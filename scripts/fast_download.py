#!/usr/bin/env python3
"""
Fast parallel Pokemon Showdown replay downloader

Downloads replays using concurrent connections for maximum speed.
Target: 100k+ replays in reasonable time.

Usage:
    python scripts/fast_download.py --formats gen1ou gen2ou --num_replays 50000 --workers 50
"""

import argparse
import asyncio
import aiohttp
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class DownloadStats:
    total_fetched: int = 0
    successful: int = 0
    failed: int = 0
    filtered_rating: int = 0
    start_time: float = 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        if self.elapsed > 0:
            return self.successful / self.elapsed
        return 0.0


class FastReplayDownloader:
    """High-performance parallel replay downloader"""

    BASE_URL = "https://replay.pokemonshowdown.com"

    def __init__(
        self,
        output_dir: str = "data/replays",
        max_concurrent: int = 50,
        min_rating: int = 1200,
        rate_limit_delay: float = 0.05,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.min_rating = min_rating
        self.rate_limit_delay = rate_limit_delay
        self.stats = DownloadStats()
        self.seen_ids: Set[str] = set()
        self.replay_queue: asyncio.Queue = None
        self.result_queue: asyncio.Queue = None

    async def fetch_replay_list(
        self,
        session: aiohttp.ClientSession,
        format_id: str,
        page: int,
    ) -> List[Dict]:
        """Fetch a page of replay metadata"""
        url = f"{self.BASE_URL}/search.json"
        params = {"format": format_id, "page": page}

        try:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return data if data else []
                elif response.status == 429:
                    # Rate limited, back off
                    await asyncio.sleep(2)
                    return []
        except Exception as e:
            pass

        return []

    async def fetch_replay(
        self,
        session: aiohttp.ClientSession,
        replay_id: str,
    ) -> Optional[Dict]:
        """Fetch a single replay's full data"""
        url = f"{self.BASE_URL}/{replay_id}.json"

        try:
            async with session.get(url, timeout=15) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    await asyncio.sleep(1)
        except Exception:
            pass

        return None

    async def worker(
        self,
        session: aiohttp.ClientSession,
        worker_id: int,
    ):
        """Worker that fetches replays from queue"""
        while True:
            try:
                replay_info = await asyncio.wait_for(
                    self.replay_queue.get(),
                    timeout=5.0,
                )

                if replay_info is None:  # Poison pill
                    break

                replay_id = replay_info["id"]

                # Small random delay to avoid hammering
                await asyncio.sleep(self.rate_limit_delay * random.uniform(0.5, 1.5))

                replay_data = await self.fetch_replay(session, replay_id)

                if replay_data:
                    replay_data["_meta"] = {
                        "rating": replay_info.get("rating", 0),
                        "format": replay_info.get("format", ""),
                    }
                    await self.result_queue.put(replay_data)
                    self.stats.successful += 1
                else:
                    self.stats.failed += 1

                self.replay_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.stats.failed += 1
                continue

    async def save_worker(self, format_id: str, batch_size: int = 1000):
        """Worker that saves replays to disk in batches"""
        batch = []
        batch_num = 0

        while True:
            try:
                replay = await asyncio.wait_for(
                    self.result_queue.get(),
                    timeout=10.0,
                )

                if replay is None:  # Poison pill
                    break

                batch.append(replay)

                if len(batch) >= batch_size:
                    self._save_batch(format_id, batch, batch_num)
                    batch_num += 1
                    batch = []

                self.result_queue.task_done()

            except asyncio.TimeoutError:
                # Save partial batch on timeout
                if batch:
                    self._save_batch(format_id, batch, batch_num)
                    batch_num += 1
                    batch = []
                continue

        # Save remaining
        if batch:
            self._save_batch(format_id, batch, batch_num)

    def _save_batch(self, format_id: str, batch: List[Dict], batch_num: int):
        """Save a batch of replays to disk"""
        filename = self.output_dir / f"{format_id}_raw_{batch_num:04d}.json"
        with open(filename, "w") as f:
            json.dump(batch, f)
        print(f"  Saved batch {batch_num}: {len(batch)} replays to {filename}")

    async def download_format(
        self,
        format_id: str,
        num_replays: int,
    ) -> int:
        """Download replays for a single format"""
        print(f"\n{'='*60}")
        print(f"Downloading {format_id} replays (target: {num_replays})")
        print(f"{'='*60}")

        self.stats = DownloadStats(start_time=time.time())
        self.replay_queue = asyncio.Queue(maxsize=self.max_concurrent * 2)
        self.result_queue = asyncio.Queue()
        self.seen_ids = set()

        # Connector with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
            ttl_dns_cache=300,
        )

        async with aiohttp.ClientSession(connector=connector) as session:
            # Start workers
            workers = [
                asyncio.create_task(self.worker(session, i))
                for i in range(self.max_concurrent)
            ]

            # Start save worker
            saver = asyncio.create_task(self.save_worker(format_id))

            # Fetch replay lists and queue them
            page = 1
            queued = 0
            empty_pages = 0

            while queued < num_replays and empty_pages < 5:
                replay_list = await self.fetch_replay_list(session, format_id, page)

                if not replay_list:
                    empty_pages += 1
                    page += 1
                    continue

                empty_pages = 0

                for replay_info in replay_list:
                    replay_id = replay_info.get("id")
                    rating = replay_info.get("rating") or 0

                    if replay_id in self.seen_ids:
                        continue

                    if rating < self.min_rating:
                        self.stats.filtered_rating += 1
                        continue

                    self.seen_ids.add(replay_id)
                    await self.replay_queue.put(replay_info)
                    queued += 1
                    self.stats.total_fetched = queued

                    if queued >= num_replays:
                        break

                page += 1

                # Progress update
                if page % 10 == 0:
                    print(
                        f"  Page {page}: queued={queued}, "
                        f"downloaded={self.stats.successful}, "
                        f"rate={self.stats.rate:.1f}/s"
                    )

                # Small delay between list fetches
                await asyncio.sleep(0.1)

            # Wait for queue to empty
            print(f"  Waiting for {self.replay_queue.qsize()} remaining downloads...")
            await self.replay_queue.join()

            # Stop workers
            for _ in workers:
                await self.replay_queue.put(None)
            await asyncio.gather(*workers)

            # Stop saver
            await self.result_queue.put(None)
            await saver

        print(f"\n{format_id} complete:")
        print(f"  Downloaded: {self.stats.successful}")
        print(f"  Failed: {self.stats.failed}")
        print(f"  Filtered (low rating): {self.stats.filtered_rating}")
        print(f"  Time: {self.stats.elapsed:.1f}s")
        print(f"  Rate: {self.stats.rate:.1f} replays/s")

        return self.stats.successful

    async def download_all(
        self,
        formats: List[str],
        num_replays_per_format: int,
    ):
        """Download replays for all formats"""
        total_start = time.time()
        total_downloaded = 0

        for format_id in formats:
            count = await self.download_format(format_id, num_replays_per_format)
            total_downloaded += count

        elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"DOWNLOAD COMPLETE")
        print(f"{'='*60}")
        print(f"Total replays: {total_downloaded}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"Average rate: {total_downloaded/elapsed:.1f} replays/s")
        print(f"Output directory: {self.output_dir}")


async def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel Pokemon Showdown replay downloader"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/replays/raw",
        help="Output directory for raw replays",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["gen1ou", "gen2ou", "gen3ou", "gen4ou", "gen5ou", "gen6ou", "gen7ou", "gen8ou", "gen9ou"],
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
        help="Minimum ELO rating filter (higher = better quality)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of concurrent download workers",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Fast Pokemon Showdown Replay Downloader")
    print("=" * 60)
    print(f"Output: {args.output_dir}")
    print(f"Formats: {args.formats}")
    print(f"Replays per format: {args.num_replays}")
    print(f"Min rating: {args.min_rating}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    downloader = FastReplayDownloader(
        output_dir=args.output_dir,
        max_concurrent=args.workers,
        min_rating=args.min_rating,
    )

    await downloader.download_all(
        formats=args.formats,
        num_replays_per_format=args.num_replays,
    )


if __name__ == "__main__":
    asyncio.run(main())
