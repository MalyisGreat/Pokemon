#!/usr/bin/env python3
"""
Download Pokemon battle datasets from Hugging Face

Available datasets:
1. metamon-parsed-replays: RL-ready trajectories (3.5M) - BEST FOR US
2. pokechamp: 2M cleaned battles (needs parsing)
3. pokemon-showdown-replays: 29M raw replays (needs parsing)

Usage:
    python scripts/download_hf_data.py --dataset metamon
    python scripts/download_hf_data.py --dataset pokechamp
    python scripts/download_hf_data.py --dataset all
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_metamon_replays(output_dir: Path, formats: list = None):
    """
    Download jakegrigsby/metamon-parsed-replays

    This is PERFECT for us - already parsed into RL trajectories!
    ~3.5M trajectories ready for training.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download, list_repo_files

    import tarfile

    repo_id = "jakegrigsby/metamon-parsed-replays"
    output_dir = output_dir / "metamon"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading from {repo_id}")
    print(f"Output: {output_dir}")

    # List available files
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        print(f"\nAvailable files ({len(files)}):")
        for f in files[:20]:
            print(f"  {f}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
    except Exception as e:
        print(f"Error listing files: {e}")
        return

    # Filter to tar.gz files (the actual data)
    data_files = [f for f in files if f.endswith('.tar.gz')]

    # Filter by format if specified
    if formats:
        data_files = [f for f in data_files if any(fmt in f for fmt in formats)]
        print(f"\nFiltered to {len(data_files)} files for formats: {formats}")
    else:
        print(f"\nFound {len(data_files)} data files to download")

    # Download and extract each file
    downloaded = 0
    for filename in data_files:
        try:
            print(f"  Downloading {filename}...", end=" ", flush=True)
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=output_dir,
            )
            print(f"OK", end=" ")

            # Extract tar.gz
            print(f"Extracting...", end=" ", flush=True)
            with tarfile.open(local_path, 'r:gz') as tar:
                tar.extractall(path=output_dir)
            print(f"Done")
            downloaded += 1
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDownloaded and extracted {downloaded} files to {output_dir}")
    return output_dir


def download_pokechamp(output_dir: Path):
    """
    Download milkkarten/pokechamp

    2M cleaned battles, needs conversion to our format.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset

    output_dir = output_dir / "pokechamp"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading milkkarten/pokechamp")
    print(f"Output: {output_dir}")
    print("This may take a while (2M+ battles)...")

    try:
        # Load dataset (streams by default)
        ds = load_dataset("milkkarten/pokechamp", split="train", streaming=True)

        # Save in batches
        batch = []
        batch_num = 0
        count = 0

        import json

        for example in ds:
            batch.append(example)
            count += 1

            if len(batch) >= 10000:
                filepath = output_dir / f"pokechamp_batch_{batch_num:04d}.json"
                with open(filepath, "w") as f:
                    json.dump(batch, f)
                print(f"  Saved batch {batch_num}: {len(batch)} battles")
                batch = []
                batch_num += 1

            if count % 100000 == 0:
                print(f"  Progress: {count} battles downloaded")

            # Optional: limit for testing
            # if count >= 100000:
            #     break

        # Save remaining
        if batch:
            filepath = output_dir / f"pokechamp_batch_{batch_num:04d}.json"
            with open(filepath, "w") as f:
                json.dump(batch, f)

        print(f"\nDownloaded {count} battles to {output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTry downloading manually:")
        print("  from datasets import load_dataset")
        print("  ds = load_dataset('milkkarten/pokechamp')")


def download_raw_replays(output_dir: Path, limit: int = 100000):
    """
    Download HolidayOugi/pokemon-showdown-replays

    29M raw replays - massive but needs parsing.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset

    output_dir = output_dir / "raw_replays"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading HolidayOugi/pokemon-showdown-replays")
    print(f"Output: {output_dir}")
    print(f"Limit: {limit} replays")
    print("WARNING: Full dataset is 29M+ replays!")

    try:
        ds = load_dataset(
            "HolidayOugi/pokemon-showdown-replays",
            split="train",
            streaming=True,
        )

        import json
        batch = []
        batch_num = 0
        count = 0

        for example in ds:
            batch.append(example)
            count += 1

            if len(batch) >= 5000:
                filepath = output_dir / f"replays_batch_{batch_num:04d}.json"
                with open(filepath, "w") as f:
                    json.dump(batch, f)
                print(f"  Saved batch {batch_num}: {len(batch)} replays")
                batch = []
                batch_num += 1

            if count >= limit:
                break

        if batch:
            filepath = output_dir / f"replays_batch_{batch_num:04d}.json"
            with open(filepath, "w") as f:
                json.dump(batch, f)

        print(f"\nDownloaded {count} replays to {output_dir}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download Pokemon datasets from HuggingFace")
    parser.add_argument(
        "--dataset",
        choices=["metamon", "pokechamp", "raw", "all"],
        default="metamon",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--output",
        default="data/hf_datasets",
        help="Output directory",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Specific formats for metamon dataset (e.g., gen1ou gen9ou)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Limit for raw replays download",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    print("=" * 60)
    print("Pokemon Dataset Downloader (Hugging Face)")
    print("=" * 60)

    if args.dataset in ["metamon", "all"]:
        download_metamon_replays(output_dir, args.formats)

    if args.dataset in ["pokechamp", "all"]:
        download_pokechamp(output_dir)

    if args.dataset in ["raw", "all"]:
        download_raw_replays(output_dir, args.limit)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print("1. Convert to training format: python scripts/convert_hf_data.py")
    print("2. Train: python scripts/train.py --config pokemon_ai/configs/h100_1gpu.yaml")


if __name__ == "__main__":
    main()
