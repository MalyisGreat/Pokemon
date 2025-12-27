#!/usr/bin/env python3
"""
Download Pokemon Showdown replays for training data

Usage:
    python scripts/download_replays.py --formats gen1ou gen2ou --num_replays 10000
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pokemon_ai.data.replay_parser import download_and_process_replays


def main():
    parser = argparse.ArgumentParser(description="Download Pokemon Showdown replays")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/replays",
        help="Output directory for replays",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["gen1ou", "gen2ou", "gen3ou", "gen4ou"],
        help="Battle formats to download",
    )
    parser.add_argument(
        "--num_replays",
        type=int,
        default=10000,
        help="Number of replays per format",
    )
    parser.add_argument(
        "--min_rating",
        type=int,
        default=1200,
        help="Minimum ELO rating filter",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Pokemon Showdown Replay Downloader")
    print("=" * 60)
    print(f"Output: {args.output_dir}")
    print(f"Formats: {args.formats}")
    print(f"Replays per format: {args.num_replays}")
    print(f"Min rating: {args.min_rating}")
    print("=" * 60)

    asyncio.run(
        download_and_process_replays(
            output_dir=args.output_dir,
            formats=args.formats,
            num_replays_per_format=args.num_replays,
            min_rating=args.min_rating,
        )
    )


if __name__ == "__main__":
    main()
