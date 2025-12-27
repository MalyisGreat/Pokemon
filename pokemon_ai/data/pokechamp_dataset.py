"""
PokeChamp Dataset - Fast loading from HuggingFace Parquet files.

This is MUCH faster than loading individual .lz4 files because:
1. Parquet is columnar and compressed efficiently
2. HuggingFace datasets library handles caching
3. Can stream directly without downloading all data first
4. Supports efficient filtering by ELO, format, etc.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from pokemon_ai.data.pokechamp_parser import PokechampBattleParser
from pokemon_ai.data.tokenizer import PokemonTokenizer
from pokemon_ai.data.dataset import PokemonDataCollator


class PokechampDataset(Dataset):
    """
    Dataset that loads from PokeChamp HuggingFace dataset.

    Much faster than .lz4 files because:
    - Parquet format is fast to load
    - Data is pre-cached by HuggingFace
    - No per-file I/O overhead
    """

    def __init__(
        self,
        split: str = "train",
        max_turns: int = 100,
        num_text_tokens: int = 87,
        gammas: Tuple[float, ...] = (0.9, 0.99, 0.999, 0.9999),
        elo_ranges: Optional[List[str]] = None,  # e.g., ["1600-1799", "1800+"]
        gamemodes: Optional[List[str]] = None,  # e.g., ["gen9ou", "gen8ou"]
        perspective: str = "winner",  # "winner", "loser", "both"
        max_samples: Optional[int] = None,
        cache_processed: bool = True,
    ):
        self.max_turns = max_turns
        self.num_text_tokens = num_text_tokens
        self.gammas = gammas
        self.perspective = perspective
        self.cache_processed = cache_processed

        # Load tokenizer and parser
        self.tokenizer = PokemonTokenizer()
        self.parser = PokechampBattleParser(perspective=perspective)

        # Load dataset from HuggingFace
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        print(f"Loading PokeChamp dataset (split={split})...")
        self.hf_dataset = load_dataset("milkkarten/pokechamp", split=split)

        # Apply filters
        if elo_ranges:
            print(f"Filtering by ELO: {elo_ranges}")
            self.hf_dataset = self.hf_dataset.filter(
                lambda x: x["elo"] in elo_ranges
            )

        if gamemodes:
            print(f"Filtering by gamemode: {gamemodes}")
            self.hf_dataset = self.hf_dataset.filter(
                lambda x: x["gamemode"] in gamemodes
            )

        if max_samples:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))

        print(f"Dataset size: {len(self.hf_dataset)} battles")

        # Cache for processed trajectories
        self._cache: Dict[int, List[Dict]] = {}

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _process_battle(self, battle_text: str) -> List[Dict[str, torch.Tensor]]:
        """Parse and process a battle into trajectory tensors"""
        # Parse battle log into turn-by-turn data
        turns = self.parser.parse(battle_text)

        if not turns:
            return []

        # Convert to tensors
        seq_len = min(len(turns), self.max_turns)
        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else 8192

        text_tokens = torch.zeros(self.max_turns, self.num_text_tokens, dtype=torch.long)
        actions = torch.zeros(self.max_turns, dtype=torch.long)
        rewards = torch.zeros(self.max_turns, dtype=torch.float)
        dones = torch.zeros(self.max_turns, dtype=torch.bool)
        turn_mask = torch.zeros(self.max_turns, dtype=torch.bool)

        for i, turn in enumerate(turns[:seq_len]):
            # Tokenize observation
            tokens = self.tokenizer.encode_observation(turn["text_obs"], self.num_text_tokens)
            tokens = tokens.clamp(0, vocab_size - 1)
            text_tokens[i] = tokens

            actions[i] = turn["action"]
            rewards[i] = turn.get("reward", 0.0)
            dones[i] = turn.get("done", False)
            turn_mask[i] = True

        # Compute returns
        returns = {}
        for gamma in self.gammas:
            ret = torch.zeros(self.max_turns, dtype=torch.float)
            running_return = 0.0
            for t in reversed(range(seq_len)):
                if dones[t]:
                    running_return = rewards[t].item()
                else:
                    running_return = rewards[t].item() + gamma * running_return
                ret[t] = running_return
            returns[gamma] = ret

        return {
            "text_tokens": text_tokens,
            "numerical_features": torch.zeros(self.max_turns, 48, dtype=torch.float),
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "turn_mask": turn_mask,
            "format_id": torch.tensor(0, dtype=torch.long),
            "won": torch.tensor(rewards[-1] > 0 if seq_len > 0 else False, dtype=torch.bool),
            "returns_0.9": returns[0.9],
            "returns_0.99": returns[0.99],
            "returns_0.999": returns[0.999],
            "returns_0.9999": returns[0.9999],
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Check cache
        if self.cache_processed and idx in self._cache:
            return self._cache[idx]

        # Get battle from HuggingFace dataset
        battle = self.hf_dataset[idx]
        battle_text = battle["text"]

        # Process
        processed = self._process_battle(battle_text)

        # If parsing failed, return empty trajectory
        if not processed:
            processed = self._empty_trajectory()

        # Cache
        if self.cache_processed:
            self._cache[idx] = processed

        return processed

    def _empty_trajectory(self) -> Dict[str, torch.Tensor]:
        """Return an empty trajectory for failed parses"""
        return {
            "text_tokens": torch.zeros(self.max_turns, self.num_text_tokens, dtype=torch.long),
            "numerical_features": torch.zeros(self.max_turns, 48, dtype=torch.float),
            "actions": torch.zeros(self.max_turns, dtype=torch.long),
            "rewards": torch.zeros(self.max_turns, dtype=torch.float),
            "dones": torch.zeros(self.max_turns, dtype=torch.bool),
            "turn_mask": torch.zeros(self.max_turns, dtype=torch.bool),
            "format_id": torch.tensor(0, dtype=torch.long),
            "won": torch.tensor(False, dtype=torch.bool),
            "returns_0.9": torch.zeros(self.max_turns, dtype=torch.float),
            "returns_0.99": torch.zeros(self.max_turns, dtype=torch.float),
            "returns_0.999": torch.zeros(self.max_turns, dtype=torch.float),
            "returns_0.9999": torch.zeros(self.max_turns, dtype=torch.float),
        }


class StreamingPokechampDataset(IterableDataset):
    """
    Streaming version that doesn't require downloading full dataset.

    Good for initial testing or when disk space is limited.
    """

    def __init__(
        self,
        split: str = "train",
        max_turns: int = 100,
        num_text_tokens: int = 87,
        gammas: Tuple[float, ...] = (0.9, 0.99, 0.999, 0.9999),
        elo_ranges: Optional[List[str]] = None,
        gamemodes: Optional[List[str]] = None,
        perspective: str = "winner",
        world_size: int = 1,
        rank: int = 0,
    ):
        self.max_turns = max_turns
        self.num_text_tokens = num_text_tokens
        self.gammas = gammas
        self.perspective = perspective
        self.world_size = world_size
        self.rank = rank
        self.elo_ranges = elo_ranges
        self.gamemodes = gamemodes
        self.split = split

        self.tokenizer = PokemonTokenizer()
        self.parser = PokechampBattleParser(perspective=perspective)

    def __iter__(self):
        from datasets import load_dataset

        # Load streaming dataset
        dataset = load_dataset("milkkarten/pokechamp", split=self.split, streaming=True)

        # Apply filters
        if self.elo_ranges:
            dataset = dataset.filter(lambda x: x["elo"] in self.elo_ranges)
        if self.gamemodes:
            dataset = dataset.filter(lambda x: x["gamemode"] in self.gamemodes)

        # Shard for distributed training
        for i, example in enumerate(dataset):
            if i % self.world_size != self.rank:
                continue

            battle_text = example["text"]
            processed = self._process_battle(battle_text)

            if processed:
                yield processed

    def _process_battle(self, battle_text: str) -> Optional[Dict[str, torch.Tensor]]:
        """Same processing as non-streaming version"""
        turns = self.parser.parse(battle_text)

        if not turns:
            return None

        seq_len = min(len(turns), self.max_turns)
        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else 8192

        text_tokens = torch.zeros(self.max_turns, self.num_text_tokens, dtype=torch.long)
        actions = torch.zeros(self.max_turns, dtype=torch.long)
        rewards = torch.zeros(self.max_turns, dtype=torch.float)
        dones = torch.zeros(self.max_turns, dtype=torch.bool)
        turn_mask = torch.zeros(self.max_turns, dtype=torch.bool)

        for i, turn in enumerate(turns[:seq_len]):
            tokens = self.tokenizer.encode_observation(turn["text_obs"], self.num_text_tokens)
            tokens = tokens.clamp(0, vocab_size - 1)
            text_tokens[i] = tokens

            actions[i] = turn["action"]
            rewards[i] = turn.get("reward", 0.0)
            dones[i] = turn.get("done", False)
            turn_mask[i] = True

        returns = {}
        for gamma in self.gammas:
            ret = torch.zeros(self.max_turns, dtype=torch.float)
            running_return = 0.0
            for t in reversed(range(seq_len)):
                if dones[t]:
                    running_return = rewards[t].item()
                else:
                    running_return = rewards[t].item() + gamma * running_return
                ret[t] = running_return
            returns[gamma] = ret

        return {
            "text_tokens": text_tokens,
            "numerical_features": torch.zeros(self.max_turns, 48, dtype=torch.float),
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "turn_mask": turn_mask,
            "format_id": torch.tensor(0, dtype=torch.long),
            "won": torch.tensor(rewards[-1] > 0 if seq_len > 0 else False, dtype=torch.bool),
            "returns_0.9": returns[0.9],
            "returns_0.99": returns[0.99],
            "returns_0.999": returns[0.999],
            "returns_0.9999": returns[0.9999],
        }


def create_pokechamp_dataloader(
    split: str = "train",
    batch_size: int = 32,
    max_turns: int = 100,
    num_workers: int = 4,
    shuffle: bool = True,
    streaming: bool = False,
    world_size: int = 1,
    rank: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    elo_ranges: Optional[List[str]] = None,
    gamemodes: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for PokeChamp dataset.

    Args:
        split: "train" or "test"
        batch_size: Batch size per GPU
        max_turns: Max turns per trajectory
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
        streaming: Use streaming (no full download needed)
        world_size: Number of GPUs
        rank: Current GPU rank
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Batches to prefetch per worker
        elo_ranges: Filter by ELO e.g., ["1600-1799", "1800+"]
        gamemodes: Filter by format e.g., ["gen9ou"]
        max_samples: Limit dataset size
    """
    if streaming:
        dataset = StreamingPokechampDataset(
            split=split,
            max_turns=max_turns,
            elo_ranges=elo_ranges,
            gamemodes=gamemodes,
            world_size=world_size,
            rank=rank,
        )
        sampler = None
    else:
        dataset = PokechampDataset(
            split=split,
            max_turns=max_turns,
            elo_ranges=elo_ranges,
            gamemodes=gamemodes,
            max_samples=max_samples,
        )

        # DistributedSampler for multi-GPU
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
        "shuffle": shuffle and sampler is None and not streaming,
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
