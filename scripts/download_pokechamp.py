#!/usr/bin/env python3
"""
Download the PokeChamp dataset from HuggingFace.

This dataset is in Parquet format which is MUCH faster to load than .lz4 files.
- 2M battles
- Already structured for ML
- Fast columnar format

Usage:
    pip install datasets
    python scripts/download_pokechamp.py --output /dev/shm/pokechamp
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download PokeChamp dataset")
    parser.add_argument("--output", type=str, default="data/pokechamp", help="Output directory")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.run(["pip", "install", "datasets"], check=True)
        from datasets import load_dataset

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading PokeChamp dataset from HuggingFace...")
    print("This contains 2M Pokemon battles in fast Parquet format")

    # Load dataset
    dataset = load_dataset("milkkarten/pokechamp")

    print(f"Dataset loaded: {dataset}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")

    # Save to disk for fast reloading
    print(f"\nSaving to {output_path}...")
    dataset.save_to_disk(str(output_path))

    print("\nDone! Dataset saved.")
    print(f"To use: dataset = load_from_disk('{output_path}')")


if __name__ == "__main__":
    main()
