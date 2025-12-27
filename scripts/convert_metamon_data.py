#!/usr/bin/env python3
"""
Convert Metamon dataset to our training format

The metamon-parsed-replays dataset is already in RL trajectory format,
we just need to adapt it to our data loader expectations.

Usage:
    python scripts/convert_metamon_data.py --input data/hf_datasets/metamon --output data/replays
"""

import argparse
import json
import lz4.frame
import sys
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def decompress_lz4(filepath: Path) -> List[Dict]:
    """Decompress lz4 compressed JSON file"""
    with lz4.frame.open(filepath, 'rb') as f:
        data = json.load(f)
    return data


def load_metamon_file(filepath: Path) -> List[Dict]:
    """Load a metamon data file (JSON or LZ4)"""
    if filepath.suffix == '.lz4':
        return decompress_lz4(filepath)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


def convert_trajectory(traj: Dict, format_id: str) -> Dict:
    """
    Convert a single Metamon trajectory to our format.

    Metamon format has:
    - observations: list of text observations
    - actions: list of action indices
    - rewards: list of rewards
    - dones: list of done flags

    Our format needs:
    - format: str
    - rating: int
    - won: bool
    - steps: list of {text_obs, numerical, action, reward, done}
    """
    steps = []

    observations = traj.get("observations", [])
    actions = traj.get("actions", [])
    rewards = traj.get("rewards", [])
    dones = traj.get("dones", [])

    # Ensure all lists have same length
    min_len = min(len(observations), len(actions), len(rewards), len(dones))

    for i in range(min_len):
        obs = observations[i]

        # Metamon obs is a string, we need to extract/create numerical features
        # For now, use dummy numerical features - the text is the main info
        step = {
            "text_obs": obs if isinstance(obs, str) else str(obs),
            "numerical": [0.0] * 48,  # Placeholder - will be filled by our parser
            "action": min(actions[i], 8),  # Clamp to valid range
            "reward": float(rewards[i]),
            "done": bool(dones[i]),
        }
        steps.append(step)

    # Determine win/loss from final reward
    won = False
    if steps and steps[-1]["reward"] > 0:
        won = True

    return {
        "format": format_id,
        "rating": traj.get("rating", 1400),  # Default rating if not provided
        "won": won,
        "steps": steps,
    }


def process_file(args) -> List[Dict]:
    """Process a single file (for multiprocessing)"""
    filepath, format_id = args
    trajectories = []

    try:
        data = load_metamon_file(filepath)

        # Data might be a list of trajectories or a dict with trajectories key
        if isinstance(data, dict):
            data = data.get("trajectories", data.get("data", [data]))

        for traj in data:
            try:
                converted = convert_trajectory(traj, format_id)
                if converted["steps"] and len(converted["steps"]) >= 3:
                    trajectories.append(converted)
            except Exception:
                continue

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

    return trajectories


def process_file_batch(args) -> List[Dict]:
    """Process a batch of files (for better CPU utilization)"""
    filepaths, format_id = args
    all_trajectories = []

    for filepath in filepaths:
        try:
            data = load_metamon_file(filepath)

            # Data might be a list of trajectories or a dict with trajectories key
            if isinstance(data, dict):
                data = data.get("trajectories", data.get("data", [data]))

            for traj in data:
                try:
                    converted = convert_trajectory(traj, format_id)
                    if converted["steps"] and len(converted["steps"]) >= 3:
                        all_trajectories.append(converted)
                except Exception:
                    continue

        except Exception:
            continue

    return all_trajectories


def extract_format_from_path(filepath: Path) -> str:
    """Extract format (e.g., gen1ou, gen2uu) from file path"""
    # Look at parent directories for format name
    # Path structure is typically: .../gen1ou/replays/... or .../gen1ou/...
    for parent in filepath.parents:
        name = parent.name.lower()
        # Match genXou, genXuu, genXnu, genXubers, genXrandombattle
        if name.startswith("gen") and len(name) >= 5:
            # Check if it's a valid format name
            if any(tier in name for tier in ["ou", "uu", "ru", "nu", "ubers", "random", "lc", "uber"]):
                return name
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Convert Metamon data to training format")
    parser.add_argument(
        "--input",
        default="data/hf_datasets/metamon",
        help="Input directory with Metamon data",
    )
    parser.add_argument(
        "--output",
        default="data/replays",
        help="Output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Files per batch (higher = more memory, better CPU util)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all data files
    files = list(input_dir.rglob("*.json")) + list(input_dir.rglob("*.lz4"))
    print(f"Found {len(files)} data files")

    if not files:
        print("No files found!")
        print(f"Make sure to download first: python scripts/download_hf_data.py --dataset metamon")
        return

    # Group by format using improved detection
    format_files = {}
    for f in files:
        format_id = extract_format_from_path(f)
        if format_id not in format_files:
            format_files[format_id] = []
        format_files[format_id].append(f)

    print(f"Formats found: {list(format_files.keys())}")

    # Process each format
    total_trajectories = 0

    for format_id, files in format_files.items():
        print(f"\nProcessing {format_id}: {len(files)} files")

        all_trajectories = []

        # Create batches for better CPU utilization
        batch_size = args.batch_size
        batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
        work_items = [(batch, format_id) for batch in batches]

        print(f"  Created {len(batches)} batches of ~{batch_size} files each")

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_file_batch, item): item for item in work_items}

            for future in tqdm(as_completed(futures), total=len(futures), desc=format_id):
                trajectories = future.result()
                all_trajectories.extend(trajectories)

        if all_trajectories:
            # Save
            output_file = output_dir / f"{format_id}_trajectories.json"
            with open(output_file, "w") as f:
                json.dump({"trajectories": all_trajectories}, f)

            print(f"  Saved {len(all_trajectories)} trajectories to {output_file}")
            total_trajectories += len(all_trajectories)

    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total trajectories: {total_trajectories}")
    print(f"Output directory: {output_dir}")
    print(f"\nNext: python scripts/train.py --config pokemon_ai/configs/h100_1gpu.yaml")


if __name__ == "__main__":
    # Check for lz4
    try:
        import lz4.frame
    except ImportError:
        print("Installing lz4...")
        import os
        os.system(f"{sys.executable} -m pip install lz4")

    main()
