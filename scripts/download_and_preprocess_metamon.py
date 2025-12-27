#!/usr/bin/env python3
"""
Download Metamon dataset from HuggingFace and preprocess to shards.

This is a one-stop script to get fast-loading training data:
1. Downloads jakegrigsby/metamon-parsed-replays tar.gz files
2. Extracts to get .lz4 files
3. Preprocesses into .pt shards for fast training

Usage:
    # On RunPod with H100s - use /dev/shm for fast I/O
    python scripts/download_and_preprocess_metamon.py \
        --output /dev/shm/metamon_shards \
        --formats gen9ou gen8ou \
        --num_workers 32

    # Full download (all formats)
    python scripts/download_and_preprocess_metamon.py \
        --output /dev/shm/metamon_shards \
        --num_workers 32
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from tqdm import tqdm

# Check dependencies
try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import hf_hub_download, list_repo_files

try:
    import lz4.frame
except ImportError:
    print("Installing lz4...")
    os.system(f"{sys.executable} -m pip install lz4")
    import lz4.frame

from pokemon_ai.data.tokenizer import PokemonTokenizer
from pokemon_ai.data.state_converter import convert_metamon_state


REPO_ID = "jakegrigsby/metamon-parsed-replays"

# Available formats in the dataset
AVAILABLE_FORMATS = [
    "gen1ou", "gen2ou", "gen3ou", "gen4ou", "gen5ou",
    "gen6ou", "gen7ou", "gen8ou", "gen9ou",
]


def list_available_files():
    """List all available tar.gz files in the repo"""
    try:
        files = list_repo_files(REPO_ID, repo_type="dataset")
        tar_files = [f for f in files if f.endswith('.tar.gz')]
        return tar_files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def download_and_extract(format_name: str, extract_dir: Path) -> Optional[Path]:
    """Download and extract a single format's tar.gz"""
    tar_filename = f"{format_name}.tar.gz"

    print(f"\n  Downloading {tar_filename}...")
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=tar_filename,
            repo_type="dataset",
        )

        # Extract
        output_dir = extract_dir / format_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Extracting to {output_dir}...")
        with tarfile.open(local_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)

        # Count extracted files
        lz4_count = len(list(output_dir.rglob("*.lz4")))
        print(f"  Extracted {lz4_count} .lz4 files")

        return output_dir
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def load_lz4_json(filepath: Path) -> Dict:
    """Load lz4-compressed JSON file"""
    with lz4.frame.open(filepath, 'rb') as f:
        return json.loads(f.read().decode('utf-8'))


def process_file(args: Tuple[str, int, int]) -> Optional[Dict[str, torch.Tensor]]:
    """Process a single .lz4 file into tensors"""
    file_path, max_turns, num_text_tokens = args

    try:
        data = load_lz4_json(Path(file_path))

        # Extract states and actions
        states = data.get("states", [])
        actions = data.get("actions", [])

        if not states or not actions:
            return None

        # Get metadata from filename
        filename = Path(file_path).name.lower()
        won = "_win" in filename

        # Parse format from filename
        format_id = "gen9ou"
        if filename.startswith("gen"):
            parts = filename.split("-")
            if parts:
                format_id = parts[0]

        # Create tokenizer (cached per process)
        if not hasattr(process_file, 'tokenizer'):
            process_file.tokenizer = PokemonTokenizer()
        tokenizer = process_file.tokenizer

        # Process trajectory
        min_len = min(len(states), len(actions))
        seq_len = min(min_len, max_turns)

        if seq_len == 0:
            return None

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

            # Actions are already correct indices in Metamon (0-8)
            actions_tensor[i] = min(actions[i], 8) if isinstance(actions[i], int) else 0
            rewards[i] = 1.0 if (i == seq_len - 1 and won) else (-1.0 if (i == seq_len - 1 and not won) else 0.0)
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


def process_chunk(args: Tuple[List[str], int, int]) -> List[Dict]:
    """Process a chunk of files"""
    file_paths, max_turns, num_text_tokens = args
    results = []
    for fp in file_paths:
        result = process_file((fp, max_turns, num_text_tokens))
        if result is not None:
            results.append(result)
    return results


def save_shard(trajectories: List[Dict], output_path: Path, shard_idx: int) -> Path:
    """Save a shard of processed trajectories"""
    shard_data = {}
    keys = trajectories[0].keys()

    for key in keys:
        tensors = [t[key] for t in trajectories]
        shard_data[key] = torch.stack(tensors)

    shard_file = output_path / f"shard_{shard_idx:05d}.pt"
    torch.save(shard_data, shard_file)
    return shard_file


def preprocess_to_shards(
    lz4_dirs: List[Path],
    output_dir: Path,
    shard_size: int = 5000,
    max_turns: int = 100,
    num_text_tokens: int = 87,
    num_workers: int = 16,
) -> Tuple[int, int]:
    """Preprocess all .lz4 files to shards"""

    # Gather all .lz4 files
    print("\nScanning for .lz4 files...")
    all_files = []
    for d in lz4_dirs:
        all_files.extend(list(d.rglob("*.lz4")))

    print(f"Found {len(all_files)} .lz4 files total")

    if not all_files:
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process in chunks for parallel efficiency
    chunk_size = max(100, len(all_files) // (num_workers * 10))
    chunks = []
    for i in range(0, len(all_files), chunk_size):
        chunk_files = [str(f) for f in all_files[i:i + chunk_size]]
        chunks.append((chunk_files, max_turns, num_text_tokens))

    print(f"Processing in {len(chunks)} chunks with {num_workers} workers...")

    total_trajectories = 0
    shard_idx = 0
    current_batch = []

    progress = tqdm(total=len(all_files), desc="Processing")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk_results in executor.map(process_chunk, chunks):
            current_batch.extend(chunk_results)
            progress.update(len(chunk_results))

            # Save shards when we have enough
            while len(current_batch) >= shard_size:
                shard_data = current_batch[:shard_size]
                current_batch = current_batch[shard_size:]

                save_shard(shard_data, output_dir, shard_idx)
                total_trajectories += len(shard_data)
                shard_idx += 1

                progress.set_postfix({"shards": shard_idx, "total": total_trajectories})

    # Save remaining
    if current_batch:
        save_shard(current_batch, output_dir, shard_idx)
        total_trajectories += len(current_batch)
        shard_idx += 1

    progress.close()

    return shard_idx, total_trajectories


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Metamon to shards")
    parser.add_argument("--output", type=str, required=True, help="Output directory for shards")
    parser.add_argument("--formats", nargs="+", default=None,
                       help=f"Formats to download (default: all). Options: {AVAILABLE_FORMATS}")
    parser.add_argument("--shard_size", type=int, default=5000, help="Trajectories per shard")
    parser.add_argument("--max_turns", type=int, default=100, help="Max turns per trajectory")
    parser.add_argument("--num_workers", type=int, default=16, help="Parallel workers for preprocessing")
    parser.add_argument("--skip_download", action="store_true", help="Skip download, only preprocess")
    parser.add_argument("--extract_dir", type=str, default=None,
                       help="Directory for extracted .lz4 files (default: temp dir)")

    args = parser.parse_args()

    formats = args.formats or AVAILABLE_FORMATS
    output_dir = Path(args.output)

    print("=" * 70)
    print("METAMON DATASET DOWNLOAD & PREPROCESS")
    print("=" * 70)
    print(f"Formats: {formats}")
    print(f"Output: {output_dir}")
    print(f"Shard size: {args.shard_size}")
    print(f"Workers: {args.num_workers}")
    print("=" * 70)

    start_time = time.time()

    # Determine extract directory
    if args.extract_dir:
        extract_dir = Path(args.extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = None
    else:
        temp_dir = tempfile.mkdtemp(prefix="metamon_")
        extract_dir = Path(temp_dir)

    extracted_dirs = []

    if not args.skip_download:
        # List available files
        print("\nListing available files on HuggingFace...")
        available = list_available_files()
        print(f"Found {len(available)} tar.gz files")

        # Download and extract each format
        print("\n" + "=" * 70)
        print("DOWNLOADING & EXTRACTING")
        print("=" * 70)

        for fmt in formats:
            tar_name = f"{fmt}.tar.gz"
            if tar_name in available:
                result = download_and_extract(fmt, extract_dir)
                if result:
                    extracted_dirs.append(result)
            else:
                print(f"  {fmt}: not found in dataset")
    else:
        # Use existing extracted directories
        for fmt in formats:
            fmt_dir = extract_dir / fmt
            if fmt_dir.exists():
                extracted_dirs.append(fmt_dir)

    if not extracted_dirs:
        print("\nNo data to process!")
        return

    # Preprocess to shards
    print("\n" + "=" * 70)
    print("PREPROCESSING TO SHARDS")
    print("=" * 70)

    num_shards, total_trajectories = preprocess_to_shards(
        extracted_dirs,
        output_dir,
        shard_size=args.shard_size,
        max_turns=args.max_turns,
        num_workers=args.num_workers,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Total shards: {num_shards}")
    print(f"Total trajectories: {total_trajectories}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Output: {output_dir}")
    print()
    print("To train with this data:")
    print(f"  python scripts/train.py --config pokemon_ai/configs/shards_2gpu.yaml \\")
    print(f"      --data_path {output_dir}")

    # Cleanup temp dir if we created one
    if temp_dir:
        print(f"\nNote: Extracted .lz4 files in temp dir: {temp_dir}")
        print("Delete manually if not needed.")


if __name__ == "__main__":
    main()
