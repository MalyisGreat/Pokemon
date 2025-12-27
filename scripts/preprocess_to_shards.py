#!/usr/bin/env python3
"""
Preprocess Metamon .lz4 files into efficient .pt shards for faster training.

This script converts millions of small .lz4 files into larger .pt shard files
that can be loaded much faster during training.

Usage:
    python scripts/preprocess_to_shards.py --input /dev/shm/metamon_data --output /dev/shm/shards --shard_size 10000

Run this in a separate terminal while training continues.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
import time

import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    print("ERROR: lz4 not installed. Run: pip install lz4")
    sys.exit(1)

from pokemon_ai.data.tokenizer import PokemonTokenizer
from pokemon_ai.data.state_converter import convert_metamon_state


def load_lz4_json(filepath: Path) -> Dict:
    """Load lz4-compressed JSON file"""
    with lz4.frame.open(filepath, 'rb') as f:
        return json.loads(f.read().decode('utf-8'))


def process_single_file(args: Tuple[str, str, int, int]) -> Dict[str, torch.Tensor]:
    """Process a single .lz4 file into tensors"""
    file_path, format_id, max_turns, num_text_tokens = args

    try:
        data = load_lz4_json(Path(file_path))

        # Extract states and actions
        states = data.get("states", [])
        actions = data.get("actions", [])

        # Get metadata from filename
        filename = Path(file_path).name.lower()
        won = "_win" in filename

        # Parse rating
        rating = 1400
        name_parts = filename.split("_")
        if len(name_parts) >= 2:
            try:
                rating = int(name_parts[1])
            except ValueError:
                pass

        # Parse format from filename
        if filename.startswith("gen"):
            parts = filename.split("-")
            if parts:
                format_id = parts[0]

        # Create tokenizer (cached per process)
        if not hasattr(process_single_file, 'tokenizer'):
            process_single_file.tokenizer = PokemonTokenizer()
        tokenizer = process_single_file.tokenizer

        # Process trajectory
        min_len = min(len(states), len(actions)) if states and actions else 0
        seq_len = min(min_len, max_turns)

        # Initialize tensors
        text_tokens = torch.zeros(max_turns, num_text_tokens, dtype=torch.long)
        actions_tensor = torch.zeros(max_turns, dtype=torch.long)
        rewards = torch.zeros(max_turns, dtype=torch.float)
        dones = torch.zeros(max_turns, dtype=torch.bool)
        turn_mask = torch.zeros(max_turns, dtype=torch.bool)

        vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else 8192

        for i in range(seq_len):
            state = states[i]
            text_obs = convert_metamon_state(state, format_id)
            tokens = tokenizer.encode_observation(text_obs, num_text_tokens)
            tokens = tokens.clamp(0, vocab_size - 1)
            text_tokens[i] = tokens

            actions_tensor[i] = min(actions[i], 8) if isinstance(actions[i], int) else 0
            rewards[i] = 1.0 if (i == seq_len - 1 and won) else 0.0
            dones[i] = i == seq_len - 1
            turn_mask[i] = True

        # Compute returns for multiple gammas
        gammas = [0.9, 0.99, 0.999, 0.9999]
        returns = {}
        for gamma in gammas:
            ret = torch.zeros(max_turns, dtype=torch.float)
            running_return = 0.0
            for t in reversed(range(seq_len)):
                if dones[t]:
                    running_return = rewards[t].item()
                else:
                    running_return = rewards[t].item() + gamma * running_return
                ret[t] = running_return
            returns[gamma] = ret

        # Format ID mapping
        format_to_id = {
            "gen1ou": 0, "gen2ou": 4, "gen3ou": 8, "gen4ou": 12,
            "gen5ou": 16, "gen6ou": 17, "gen7ou": 18, "gen8ou": 19, "gen9ou": 20,
        }
        format_id_num = format_to_id.get(format_id.lower(), 0)

        return {
            "text_tokens": text_tokens,
            "numerical_features": torch.zeros(max_turns, 48, dtype=torch.float),
            "actions": actions_tensor,
            "rewards": rewards,
            "dones": dones,
            "turn_mask": turn_mask,
            "format_id": torch.tensor(format_id_num, dtype=torch.long),
            "won": torch.tensor(won, dtype=torch.bool),
            "returns_0.9": returns[0.9],
            "returns_0.99": returns[0.99],
            "returns_0.999": returns[0.999],
            "returns_0.9999": returns[0.9999],
        }
    except Exception as e:
        return None


def process_batch(file_paths: List[str], max_turns: int, num_text_tokens: int, num_workers: int) -> List[Dict]:
    """Process a batch of files in parallel"""
    args_list = [(fp, "gen9ou", max_turns, num_text_tokens) for fp in file_paths]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_file, args) for args in args_list]
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)

    return results


def save_shard(trajectories: List[Dict], output_path: Path, shard_idx: int):
    """Save a shard of processed trajectories"""
    if not trajectories:
        return

    # Stack all tensors
    shard_data = {}
    keys = trajectories[0].keys()

    for key in keys:
        tensors = [t[key] for t in trajectories]
        shard_data[key] = torch.stack(tensors)

    shard_file = output_path / f"shard_{shard_idx:05d}.pt"
    torch.save(shard_data, shard_file)
    return shard_file


def main():
    parser = argparse.ArgumentParser(description="Preprocess Metamon data to .pt shards")
    parser.add_argument("--input", type=str, required=True, help="Input directory with .lz4 files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for .pt shards")
    parser.add_argument("--shard_size", type=int, default=10000, help="Trajectories per shard")
    parser.add_argument("--max_turns", type=int, default=100, help="Max turns per trajectory")
    parser.add_argument("--num_text_tokens", type=int, default=87, help="Text tokens per turn")
    parser.add_argument("--num_workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to process (for testing)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for .lz4 files in {input_path}...")
    start_time = time.time()

    # Find all .lz4 files
    lz4_files = list(input_path.rglob("*.lz4"))
    print(f"Found {len(lz4_files)} .lz4 files in {time.time() - start_time:.1f}s")

    if args.max_files:
        lz4_files = lz4_files[:args.max_files]
        print(f"Limited to {len(lz4_files)} files for testing")

    # Process in batches
    total_trajectories = 0
    shard_idx = 0
    current_batch = []

    batch_size = args.num_workers * 4  # Process multiple per worker

    progress = tqdm(total=len(lz4_files), desc="Processing files")

    for i in range(0, len(lz4_files), batch_size):
        batch_files = [str(f) for f in lz4_files[i:i + batch_size]]

        # Process batch
        results = process_batch(batch_files, args.max_turns, args.num_text_tokens, args.num_workers)
        current_batch.extend(results)

        progress.update(len(batch_files))

        # Save shard when full
        while len(current_batch) >= args.shard_size:
            shard_data = current_batch[:args.shard_size]
            current_batch = current_batch[args.shard_size:]

            shard_file = save_shard(shard_data, output_path, shard_idx)
            total_trajectories += len(shard_data)
            shard_idx += 1

            progress.set_postfix({
                "shards": shard_idx,
                "trajectories": total_trajectories,
            })

    # Save remaining
    if current_batch:
        save_shard(current_batch, output_path, shard_idx)
        total_trajectories += len(current_batch)
        shard_idx += 1

    progress.close()

    elapsed = time.time() - start_time
    print(f"\nDone!")
    print(f"  Total shards: {shard_idx}")
    print(f"  Total trajectories: {total_trajectories}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Speed: {len(lz4_files) / elapsed:.0f} files/sec")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
