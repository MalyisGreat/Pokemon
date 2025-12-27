#!/usr/bin/env python3
"""
Process raw replays into training trajectories

Usage:
    python scripts/process_replays.py --input data/replays/raw --output data/replays
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pokemon_ai.data.replay_parser import ReplayParser


def process_replay_file(args) -> List[Dict]:
    """Process a single replay file (runs in separate process)"""
    filepath, output_dir = args
    parser = ReplayParser(output_dir=output_dir)
    trajectories = []

    try:
        with open(filepath, "r") as f:
            replays = json.load(f)

        for replay in replays:
            # Process from both players' perspectives
            for pov in [1, 2]:
                try:
                    traj = parser.parse_replay(replay, pov_player=pov)
                    if traj["steps"] and len(traj["steps"]) >= 5:  # Minimum 5 turns
                        trajectories.append(traj)
                except Exception:
                    continue

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Process raw replays into training data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/replays/raw",
        help="Input directory with raw replay JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/replays",
        help="Output directory for processed trajectories",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--min_turns",
        type=int,
        default=5,
        help="Minimum turns per trajectory",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all raw replay files
    replay_files = list(input_dir.glob("*.json"))
    print(f"Found {len(replay_files)} replay files to process")

    if not replay_files:
        print("No replay files found!")
        return

    # Group files by format
    format_files = {}
    for f in replay_files:
        # Extract format from filename (e.g., gen1ou_raw_0001.json -> gen1ou)
        parts = f.stem.split("_")
        if len(parts) >= 2:
            format_id = parts[0]
            if format_id not in format_files:
                format_files[format_id] = []
            format_files[format_id].append(f)

    # Process each format
    for format_id, files in format_files.items():
        print(f"\nProcessing {format_id}: {len(files)} files")

        all_trajectories = []

        # Process in parallel
        work_items = [(f, str(output_dir)) for f in files]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_replay_file, item): item
                for item in work_items
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=format_id):
                trajectories = future.result()
                all_trajectories.extend(trajectories)

        # Save processed trajectories
        output_file = output_dir / f"{format_id}_trajectories.json"
        with open(output_file, "w") as f:
            json.dump({"trajectories": all_trajectories}, f)

        print(f"  Saved {len(all_trajectories)} trajectories to {output_file}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
