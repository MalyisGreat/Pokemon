#!/usr/bin/env python3
"""
Download ALL available high-quality Pokemon Showdown replays

This script maximizes data collection by:
1. Downloading from all common formats
2. Using multiple rating thresholds (prioritizing high rating)
3. Running in parallel with rate limiting
4. Saving incrementally to avoid data loss

Target: 500k-2M trajectories for offline RL training

Usage:
    python scripts/download_all_data.py --quality high
    python scripts/download_all_data.py --quality medium --workers 100
"""

import argparse
import asyncio
import aiohttp
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass, field
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# All major formats with their typical replay availability
FORMATS = {
    # Current gen (most replays)
    "gen9ou": {"priority": 1, "expected_replays": 100000},
    "gen9uu": {"priority": 2, "expected_replays": 50000},
    "gen9ru": {"priority": 3, "expected_replays": 30000},
    "gen9nu": {"priority": 3, "expected_replays": 20000},
    "gen9monotype": {"priority": 3, "expected_replays": 20000},
    "gen9randombattle": {"priority": 2, "expected_replays": 100000},
    "gen9doublesou": {"priority": 3, "expected_replays": 30000},

    # Previous gens (good data, established metagame)
    "gen8ou": {"priority": 1, "expected_replays": 80000},
    "gen8uu": {"priority": 2, "expected_replays": 40000},
    "gen8randombattle": {"priority": 2, "expected_replays": 80000},

    "gen7ou": {"priority": 1, "expected_replays": 60000},
    "gen7uu": {"priority": 3, "expected_replays": 20000},
    "gen7randombattle": {"priority": 2, "expected_replays": 50000},

    "gen6ou": {"priority": 2, "expected_replays": 40000},
    "gen5ou": {"priority": 2, "expected_replays": 30000},
    "gen4ou": {"priority": 2, "expected_replays": 25000},
    "gen3ou": {"priority": 2, "expected_replays": 20000},
    "gen2ou": {"priority": 3, "expected_replays": 15000},
    "gen1ou": {"priority": 3, "expected_replays": 15000},
}

QUALITY_PRESETS = {
    "high": {
        "min_rating": 1500,
        "description": "High quality games (1500+ ELO)",
    },
    "medium": {
        "min_rating": 1300,
        "description": "Medium quality games (1300+ ELO)",
    },
    "all": {
        "min_rating": 1000,
        "description": "All reasonable games (1000+ ELO)",
    },
}


@dataclass
class DownloadProgress:
    format_id: str
    target: int
    downloaded: int = 0
    failed: int = 0
    filtered: int = 0
    pages_scanned: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.downloaded / elapsed if elapsed > 0 else 0


class MassReplayDownloader:
    """Download as many replays as possible"""

    BASE_URL = "https://replay.pokemonshowdown.com"

    def __init__(
        self,
        output_dir: str,
        min_rating: int = 1300,
        max_concurrent: int = 50,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_rating = min_rating
        self.max_concurrent = max_concurrent
        self.seen_ids: Set[str] = set()
        self.total_downloaded = 0
        self.total_failed = 0

    async def fetch_with_retry(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict = None,
        max_retries: int = 3,
    ) -> dict | list | None:
        """Fetch URL with retry logic"""
        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        return None
            except Exception:
                await asyncio.sleep(1)
        return None

    async def download_format(
        self,
        session: aiohttp.ClientSession,
        format_id: str,
        max_replays: int = 50000,
    ) -> int:
        """Download replays for a single format"""
        progress = DownloadProgress(format_id=format_id, target=max_replays)
        replays_batch = []
        batch_num = 0
        page = 1
        empty_streak = 0

        print(f"\n  [{format_id}] Starting download (target: {max_replays})")

        while progress.downloaded < max_replays and empty_streak < 10:
            # Fetch page of replay metadata
            data = await self.fetch_with_retry(
                session,
                f"{self.BASE_URL}/search.json",
                {"format": format_id, "page": page},
            )

            if not data:
                empty_streak += 1
                page += 1
                continue

            empty_streak = 0
            progress.pages_scanned += 1

            # Queue up replay downloads
            tasks = []
            for replay_info in data:
                replay_id = replay_info.get("id")
                rating = replay_info.get("rating") or 0

                if replay_id in self.seen_ids:
                    continue

                if rating < self.min_rating:
                    progress.filtered += 1
                    continue

                self.seen_ids.add(replay_id)
                tasks.append(self.fetch_replay(session, replay_id))

            # Download in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, dict) and result:
                        replays_batch.append(result)
                        progress.downloaded += 1
                        self.total_downloaded += 1
                    else:
                        progress.failed += 1
                        self.total_failed += 1

            # Save batch periodically
            if len(replays_batch) >= 500:
                self._save_batch(format_id, replays_batch, batch_num)
                batch_num += 1
                replays_batch = []

            # Progress update
            if page % 20 == 0:
                print(
                    f"  [{format_id}] Page {page}: "
                    f"{progress.downloaded}/{max_replays} "
                    f"({progress.rate:.1f}/s)"
                )

            page += 1
            await asyncio.sleep(0.05)  # Rate limit

        # Save remaining
        if replays_batch:
            self._save_batch(format_id, replays_batch, batch_num)

        elapsed = time.time() - progress.start_time
        print(
            f"  [{format_id}] Complete: {progress.downloaded} replays "
            f"in {elapsed:.0f}s ({progress.rate:.1f}/s)"
        )

        return progress.downloaded

    async def fetch_replay(
        self,
        session: aiohttp.ClientSession,
        replay_id: str,
    ) -> dict | None:
        """Fetch single replay"""
        await asyncio.sleep(random.uniform(0.01, 0.05))
        return await self.fetch_with_retry(
            session,
            f"{self.BASE_URL}/{replay_id}.json",
        )

    def _save_batch(self, format_id: str, batch: list, batch_num: int):
        """Save batch to disk"""
        filepath = self.output_dir / f"{format_id}_raw_{batch_num:04d}.json"
        with open(filepath, "w") as f:
            json.dump(batch, f)
        print(f"  [{format_id}] Saved {len(batch)} replays to {filepath.name}")

    async def download_all(
        self,
        formats: list[str] = None,
        max_per_format: int = 20000,
    ):
        """Download from all formats"""
        formats = formats or list(FORMATS.keys())

        print("=" * 60)
        print("MASS REPLAY DOWNLOAD")
        print("=" * 60)
        print(f"Formats: {len(formats)}")
        print(f"Min rating: {self.min_rating}")
        print(f"Max per format: {max_per_format}")
        print(f"Estimated total: {len(formats) * max_per_format}")
        print("=" * 60)

        start = time.time()

        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
        )

        async with aiohttp.ClientSession(connector=connector) as session:
            for format_id in formats:
                await self.download_format(session, format_id, max_per_format)

        elapsed = time.time() - start
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"Total downloaded: {self.total_downloaded}")
        print(f"Total failed: {self.total_failed}")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Rate: {self.total_downloaded/elapsed:.1f} replays/s")
        print(f"Output: {self.output_dir}")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Download all Pokemon Showdown replays")
    parser.add_argument(
        "--output",
        default="data/replays/raw",
        help="Output directory",
    )
    parser.add_argument(
        "--quality",
        choices=["high", "medium", "all"],
        default="medium",
        help="Quality preset (high=1500+, medium=1300+, all=1000+)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Specific formats to download (default: all)",
    )
    parser.add_argument(
        "--max_per_format",
        type=int,
        default=20000,
        help="Max replays per format",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Concurrent workers",
    )

    args = parser.parse_args()

    preset = QUALITY_PRESETS[args.quality]
    print(f"Quality: {args.quality} - {preset['description']}")

    downloader = MassReplayDownloader(
        output_dir=args.output,
        min_rating=preset["min_rating"],
        max_concurrent=args.workers,
    )

    await downloader.download_all(
        formats=args.formats,
        max_per_format=args.max_per_format,
    )


if __name__ == "__main__":
    asyncio.run(main())
