"""
Fast dataset for loading preprocessed .pt shards.

Each shard contains ~10k trajectories pre-tokenized and ready to use.
This is 10-100x faster than loading individual .lz4 files.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from pokemon_ai.data.dataset import PokemonDataCollator


class ShardDataset(Dataset):
    """
    Dataset that loads from preprocessed .pt shards.

    Each shard is a dict of stacked tensors:
    - text_tokens: [shard_size, max_turns, num_text_tokens]
    - actions: [shard_size, max_turns]
    - rewards: [shard_size, max_turns]
    - etc.

    This is much faster than loading individual .lz4 files because:
    1. Fewer file opens (400 shards vs 4M files)
    2. No decompression needed
    3. No JSON parsing
    4. No tokenization
    5. Tensors are already in the right format
    """

    def __init__(
        self,
        shard_dir: str,
        max_turns: int = 100,
        gammas: Tuple[float, ...] = (0.9, 0.99, 0.999, 0.9999),
        shuffle_shards: bool = True,
        preload_all: bool = False,
    ):
        self.shard_dir = Path(shard_dir)
        self.max_turns = max_turns
        self.gammas = gammas
        self.shuffle_shards = shuffle_shards
        self.preload_all = preload_all

        # Find all shard files
        self.shard_files = sorted(self.shard_dir.glob("shard_*.pt"))
        if not self.shard_files:
            raise ValueError(f"No shard files found in {shard_dir}")

        print(f"Found {len(self.shard_files)} shards")

        # Load first shard to get shard size
        first_shard = torch.load(self.shard_files[0])
        self.shard_size = first_shard["text_tokens"].shape[0]

        # Build index: (shard_idx, local_idx) for each trajectory
        self.total_trajectories = len(self.shard_files) * self.shard_size

        # For last shard, check actual size
        last_shard = torch.load(self.shard_files[-1])
        last_shard_size = last_shard["text_tokens"].shape[0]
        if last_shard_size < self.shard_size:
            self.total_trajectories = (len(self.shard_files) - 1) * self.shard_size + last_shard_size

        print(f"Total trajectories: {self.total_trajectories}")

        # Cache for loaded shards
        self._shard_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cache_order: List[int] = []
        self._max_cache_size = 5  # Keep 5 shards in memory

        # Optionally preload all shards
        if preload_all:
            print("Preloading all shards to RAM...")
            for i, shard_file in enumerate(self.shard_files):
                self._shard_cache[i] = torch.load(shard_file)
            print(f"Preloaded {len(self._shard_cache)} shards")

    def _load_shard(self, shard_idx: int) -> Dict[str, torch.Tensor]:
        """Load a shard with LRU caching"""
        if shard_idx in self._shard_cache:
            return self._shard_cache[shard_idx]

        # Load shard
        shard_data = torch.load(self.shard_files[shard_idx])

        # Add to cache
        self._shard_cache[shard_idx] = shard_data
        self._cache_order.append(shard_idx)

        # Evict oldest if cache too large
        while len(self._cache_order) > self._max_cache_size:
            oldest = self._cache_order.pop(0)
            if oldest in self._shard_cache and oldest != shard_idx:
                del self._shard_cache[oldest]

        return shard_data

    def __len__(self) -> int:
        return self.total_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which shard this index belongs to
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size

        # Handle last shard potentially being smaller
        if shard_idx >= len(self.shard_files):
            shard_idx = len(self.shard_files) - 1
            local_idx = idx - shard_idx * self.shard_size

        # Load shard
        shard_data = self._load_shard(shard_idx)

        # Check bounds for last shard
        if local_idx >= shard_data["text_tokens"].shape[0]:
            local_idx = shard_data["text_tokens"].shape[0] - 1

        # Extract this trajectory
        return {
            "text_tokens": shard_data["text_tokens"][local_idx],
            "numerical_features": shard_data["numerical_features"][local_idx],
            "actions": shard_data["actions"][local_idx],
            "rewards": shard_data["rewards"][local_idx],
            "dones": shard_data["dones"][local_idx],
            "turn_mask": shard_data["turn_mask"][local_idx],
            "format_id": shard_data["format_id"][local_idx],
            "won": shard_data["won"][local_idx],
            "returns_0.9": shard_data["returns_0.9"][local_idx],
            "returns_0.99": shard_data["returns_0.99"][local_idx],
            "returns_0.999": shard_data["returns_0.999"][local_idx],
            "returns_0.9999": shard_data["returns_0.9999"][local_idx],
        }


def create_shard_dataloader(
    shard_dir: str,
    batch_size: int = 32,
    max_turns: int = 100,
    num_workers: int = 4,
    shuffle: bool = True,
    world_size: int = 1,
    rank: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    preload_all: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for shard-based data.

    This is much faster than the regular dataloader because:
    1. No per-item file I/O
    2. No decompression
    3. No tokenization
    4. Shards are cached in memory
    """
    dataset = ShardDataset(
        shard_dir=shard_dir,
        max_turns=max_turns,
        preload_all=preload_all,
    )

    # Use DistributedSampler for multi-GPU
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )

    collator = PokemonDataCollator(max_turns=max_turns)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory,
        "drop_last": True,
        "persistent_workers": num_workers > 0,
    }

    if sampler is not None:
        loader_kwargs["sampler"] = sampler

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)
