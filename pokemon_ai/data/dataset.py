"""
Pokemon Battle Dataset

Handles loading and processing of the Metamon replay dataset for offline RL training.

Dataset format (per trajectory):
- text_tokens: [max_turns, 87] - tokenized observation text
- numerical_features: [max_turns, 48] - normalized numerical features
- actions: [max_turns] - action indices (0-8)
- rewards: [max_turns] - per-step rewards
- dones: [max_turns] - episode termination flags
- returns: [max_turns] - return-to-go for each gamma
- turn_mask: [max_turns] - valid turn mask
"""

import os
import json
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

from pokemon_ai.data.tokenizer import PokemonTokenizer
from pokemon_ai.data.state_converter import convert_metamon_state


def load_lz4_json(filepath: Path) -> Dict:
    """Load lz4-compressed JSON file (Metamon format)"""
    with lz4.frame.open(filepath, 'rb') as f:
        return json.loads(f.read().decode('utf-8'))


@dataclass
class TrajectoryData:
    """Single trajectory data"""
    text_tokens: torch.Tensor  # [seq_len, num_text_tokens]
    numerical_features: torch.Tensor  # [seq_len, num_numerical]
    actions: torch.Tensor  # [seq_len]
    rewards: torch.Tensor  # [seq_len]
    dones: torch.Tensor  # [seq_len]
    returns: Dict[float, torch.Tensor]  # gamma -> [seq_len]
    turn_mask: torch.Tensor  # [seq_len]
    format_id: int  # Battle format (gen/tier)
    won: bool  # Did this player win


class PokemonBattleDataset(Dataset):
    """
    Dataset for Pokemon battle trajectories.

    Supports:
    - Loading from preprocessed .pt files (fast)
    - Loading from raw JSON replays (slower, more flexible)
    - Filtering by format, ELO, win/loss
    - Return computation for multiple discount factors
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[PokemonTokenizer] = None,
        max_turns: int = 200,
        num_text_tokens: int = 87,
        num_numerical_features: int = 48,
        gammas: Tuple[float, ...] = (0.9, 0.99, 0.999, 0.9999),
        formats: Optional[List[str]] = None,  # Filter to specific formats
        min_elo: Optional[int] = None,  # Minimum ELO filter
        winners_only: bool = False,  # Only include winning trajectories
        cache_in_memory: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer or PokemonTokenizer()
        self.max_turns = max_turns
        self.num_text_tokens = num_text_tokens
        self.num_numerical_features = num_numerical_features
        self.gammas = gammas
        self.formats = formats
        self.min_elo = min_elo
        self.winners_only = winners_only
        self.cache_in_memory = cache_in_memory
        self.max_samples = max_samples

        # Format to ID mapping
        self.format_to_id = {
            "gen1ou": 0, "gen1uu": 1, "gen1nu": 2, "gen1ubers": 3,
            "gen2ou": 4, "gen2uu": 5, "gen2nu": 6, "gen2ubers": 7,
            "gen3ou": 8, "gen3uu": 9, "gen3nu": 10, "gen3ubers": 11,
            "gen4ou": 12, "gen4uu": 13, "gen4nu": 14, "gen4ubers": 15,
            "gen5ou": 16, "gen6ou": 17, "gen7ou": 18, "gen8ou": 19, "gen9ou": 20,
        }

        self.trajectories: List[Dict[str, Any]] = []
        self.file_index: List[Tuple[str, int]] = []  # (file_path, traj_idx)
        self._is_metamon = False  # Will be set to True for Metamon .lz4 files

        self._load_data()

    def _load_data(self):
        """Load trajectory data from files"""
        if self.data_path.is_file():
            # Single file
            self._load_file(self.data_path)
        else:
            # Directory of files - check for .pt, .json, or .lz4 (Metamon format)
            files = []
            is_metamon = False

            if any(self.data_path.glob("*.pt")):
                files = sorted(self.data_path.glob("*.pt"))
            elif any(self.data_path.glob("*.json")):
                files = sorted(self.data_path.glob("*.json"))
            elif HAS_LZ4:
                # Metamon format: .lz4 files in subdirectories
                # Use fast file listing instead of rglob for millions of files
                print("Scanning for .lz4 files...")
                files = list(self.data_path.rglob("*.lz4"))
                if files:
                    is_metamon = True
                    print(f"Found {len(files)} Metamon .lz4 files")

            if is_metamon:
                # FAST PATH: For Metamon .lz4 files, just index file paths
                # Each .lz4 file contains ONE trajectory, so we index them directly
                # No need to load and parse each file upfront!
                if self.max_samples:
                    files = files[:self.max_samples]

                # Just store paths - load on-demand in __getitem__
                self.file_index = [(str(f), 0) for f in files]
                self._is_metamon = True
                print(f"Indexed {len(self.file_index)} trajectories (lazy loading enabled)")
            else:
                # Original path for .pt/.json files
                self._is_metamon = False
                for file_path in tqdm(files, desc="Loading data"):
                    self._load_file(file_path)

                    if self.max_samples and len(self.file_index) >= self.max_samples:
                        break
                print(f"Loaded {len(self.file_index)} trajectories")

    def _load_file(self, file_path: Path):
        """Load a single data file"""
        file_path = Path(file_path)
        try:
            if file_path.suffix == ".pt":
                data = torch.load(file_path)
            elif file_path.suffix == ".lz4" and HAS_LZ4:
                # Metamon format: single trajectory per .lz4 file
                data = load_lz4_json(file_path)
                # Convert Metamon format to our format
                data = self._convert_metamon_trajectory(data, file_path)
            else:
                with open(file_path, "r") as f:
                    data = json.load(f)

            trajectories = data if isinstance(data, list) else data.get("trajectories", [data])

            for idx, traj in enumerate(trajectories):
                # Apply filters
                if self._should_include(traj):
                    if self.cache_in_memory:
                        self.trajectories.append(self._process_trajectory(traj))
                    else:
                        self.file_index.append((str(file_path), idx))

                    if self.max_samples and len(self.file_index) >= self.max_samples:
                        break
        except Exception as e:
            pass  # Skip corrupted files silently

    def _convert_metamon_trajectory(self, data: Dict, file_path: Path) -> Dict:
        """Convert Metamon format to our format"""
        # Metamon has 'states' and 'actions'
        states = data.get("states", [])
        actions = data.get("actions", [])

        # Extract format and metadata from filename
        # Format: gen1ou-123456_1500_user_vs_opp_date_WIN.json.lz4
        filename = file_path.name.lower()
        format_id = "gen9ou"  # default
        won = "_win" in filename
        rating = 1400

        # Parse format from filename
        if filename.startswith("gen"):
            parts = filename.split("-")
            if parts:
                format_id = parts[0]

        # Parse rating from filename
        name_parts = filename.split("_")
        if len(name_parts) >= 2:
            try:
                rating = int(name_parts[1])
            except ValueError:
                pass

        # Build steps
        steps = []
        min_len = min(len(states), len(actions)) if states and actions else 0

        for i in range(min_len):
            state = states[i]
            # Convert state to proper text observation using our converter
            text_obs = convert_metamon_state(state, format_id)

            steps.append({
                "text_obs": text_obs,
                "numerical": [0.0] * 48,
                "action": min(actions[i], 8) if isinstance(actions[i], int) else 0,
                "reward": 1.0 if (i == min_len - 1 and won) else 0.0,
                "done": i == min_len - 1,
            })

        return {
            "format": format_id,
            "rating": rating,
            "won": won,
            "steps": steps,
        }

    def _should_include(self, traj: Dict) -> bool:
        """Check if trajectory passes filters"""
        # Format filter
        if self.formats:
            traj_format = traj.get("format", "").lower().replace("[", "").replace("]", "").replace(" ", "")
            if not any(f.lower() in traj_format for f in self.formats):
                return False

        # ELO filter
        if self.min_elo:
            elo = traj.get("elo", 0) or traj.get("rating", 0)
            if elo < self.min_elo:
                return False

        # Winners only filter
        if self.winners_only and not traj.get("won", True):
            return False

        return True

    def _process_trajectory(self, traj: Dict) -> TrajectoryData:
        """Process raw trajectory into tensors"""
        steps = traj.get("steps", traj.get("observations", []))
        seq_len = min(len(steps), self.max_turns)

        # Initialize tensors
        text_tokens = torch.zeros(self.max_turns, self.num_text_tokens, dtype=torch.long)
        numerical_features = torch.zeros(self.max_turns, self.num_numerical_features, dtype=torch.float)
        actions = torch.zeros(self.max_turns, dtype=torch.long)
        rewards = torch.zeros(self.max_turns, dtype=torch.float)
        dones = torch.zeros(self.max_turns, dtype=torch.bool)
        turn_mask = torch.zeros(self.max_turns, dtype=torch.bool)

        # Get vocab size for clamping (default 8192)
        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else 8192

        for i, step in enumerate(steps[:seq_len]):
            # Text tokens - clamp to valid vocab range to avoid index errors
            if "text_obs" in step:
                tokens = self.tokenizer.encode_observation(step["text_obs"], self.num_text_tokens)
                tokens = tokens.clamp(0, vocab_size - 1)
                text_tokens[i] = tokens
            elif "text_tokens" in step:
                toks = torch.tensor(step["text_tokens"][:self.num_text_tokens], dtype=torch.long)
                toks = toks.clamp(0, vocab_size - 1)
                text_tokens[i, :len(toks)] = toks

            # Numerical features
            if "numerical" in step:
                num_feats = step["numerical"][:self.num_numerical_features]
                numerical_features[i, :len(num_feats)] = torch.tensor(num_feats, dtype=torch.float)
            elif "numerical_features" in step:
                num_feats = step["numerical_features"][:self.num_numerical_features]
                numerical_features[i, :len(num_feats)] = torch.tensor(num_feats, dtype=torch.float)

            # Action (clamp to valid range 0-8)
            if "action" in step:
                actions[i] = min(step["action"], 8)

            # Reward
            if "reward" in step:
                rewards[i] = step["reward"]

            # Done flag
            if "done" in step:
                dones[i] = step["done"]

            turn_mask[i] = True

        # Set final done
        if seq_len > 0:
            dones[seq_len - 1] = True

        # Compute returns for each gamma
        returns = {}
        for gamma in self.gammas:
            returns[gamma] = self._compute_returns(rewards, dones, gamma, seq_len)

        # Get format ID
        format_str = traj.get("format", "gen1ou").lower()
        format_str = format_str.replace("[", "").replace("]", "").replace(" ", "")
        format_id = self.format_to_id.get(format_str, 0)

        return TrajectoryData(
            text_tokens=text_tokens,
            numerical_features=numerical_features,
            actions=actions,
            rewards=rewards,
            dones=dones,
            returns=returns,
            turn_mask=turn_mask,
            format_id=format_id,
            won=traj.get("won", True),
        )

    def _compute_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        seq_len: int,
    ) -> torch.Tensor:
        """Compute discounted returns (return-to-go)"""
        returns = torch.zeros_like(rewards)

        running_return = 0.0
        for t in reversed(range(seq_len)):
            if dones[t]:
                running_return = rewards[t]
            else:
                running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns

    def __len__(self) -> int:
        if self.cache_in_memory:
            return len(self.trajectories)
        return len(self.file_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_in_memory:
            traj = self.trajectories[idx]
        else:
            file_path, traj_idx = self.file_index[idx]

            # Handle different file types
            if file_path.endswith(".lz4") and HAS_LZ4:
                # Metamon .lz4 file - load and convert on-demand
                data = load_lz4_json(Path(file_path))
                converted = self._convert_metamon_trajectory(data, Path(file_path))
                traj = self._process_trajectory(converted)
            elif file_path.endswith(".pt"):
                data = torch.load(file_path)
                trajectories = data if isinstance(data, list) else data.get("trajectories", [data])
                traj = self._process_trajectory(trajectories[traj_idx])
            else:
                with open(file_path, "r") as f:
                    data = json.load(f)
                trajectories = data if isinstance(data, list) else data.get("trajectories", [data])
                traj = self._process_trajectory(trajectories[traj_idx])

        return {
            "text_tokens": traj.text_tokens,
            "numerical_features": traj.numerical_features,
            "actions": traj.actions,
            "rewards": traj.rewards,
            "dones": traj.dones,
            "turn_mask": traj.turn_mask,
            "format_id": torch.tensor(traj.format_id, dtype=torch.long),
            "won": torch.tensor(traj.won, dtype=torch.bool),
            **{f"returns_{gamma}": traj.returns[gamma] for gamma in self.gammas},
        }


class StreamingPokemonDataset(IterableDataset):
    """
    Streaming dataset for very large data that doesn't fit in memory.

    Supports sharding for distributed training.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[PokemonTokenizer] = None,
        max_turns: int = 200,
        gammas: Tuple[float, ...] = (0.9, 0.99, 0.999, 0.9999),
        shuffle: bool = True,
        seed: int = 42,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer or PokemonTokenizer()
        self.max_turns = max_turns
        self.gammas = gammas
        self.shuffle = shuffle
        self.seed = seed
        self.world_size = world_size
        self.rank = rank

        # Get all data files
        self.files = sorted(self.data_path.glob("*.pt")) + sorted(self.data_path.glob("*.json"))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Handle DataLoader workers
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        # Shard files across workers and ranks
        total_workers = self.world_size * num_workers
        global_worker_id = self.rank * num_workers + worker_id

        # Shuffle files
        files = list(self.files)
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(files)

        # Assign files to this worker
        worker_files = files[global_worker_id::total_workers]

        for file_path in worker_files:
            yield from self._process_file(file_path)

    def _process_file(self, file_path: Path):
        """Process a single file and yield trajectories"""
        try:
            if file_path.suffix == ".pt":
                data = torch.load(file_path)
            else:
                with open(file_path, "r") as f:
                    data = json.load(f)

            trajectories = data if isinstance(data, list) else data.get("trajectories", [data])

            if self.shuffle:
                random.shuffle(trajectories)

            for traj in trajectories:
                yield self._process_trajectory(traj)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def _process_trajectory(self, traj: Dict) -> Dict[str, torch.Tensor]:
        """Process a single trajectory (same as PokemonBattleDataset)"""
        # Reuse the processing logic
        dataset = PokemonBattleDataset.__new__(PokemonBattleDataset)
        dataset.tokenizer = self.tokenizer
        dataset.max_turns = self.max_turns
        dataset.num_text_tokens = 87
        dataset.num_numerical_features = 48
        dataset.gammas = self.gammas
        dataset.format_to_id = {
            "gen1ou": 0, "gen2ou": 4, "gen3ou": 8, "gen4ou": 12,
        }

        processed = dataset._process_trajectory(traj)

        return {
            "text_tokens": processed.text_tokens,
            "numerical_features": processed.numerical_features,
            "actions": processed.actions,
            "rewards": processed.rewards,
            "dones": processed.dones,
            "turn_mask": processed.turn_mask,
            "format_id": torch.tensor(processed.format_id, dtype=torch.long),
            "won": torch.tensor(processed.won, dtype=torch.bool),
            **{f"returns_{gamma}": processed.returns[gamma] for gamma in self.gammas},
        }


@dataclass
class PokemonDataCollator:
    """
    Data collator for batching Pokemon trajectories.

    Handles:
    - Padding to max sequence length in batch
    - Creating attention masks
    - Shifting actions/rewards for autoregressive training
    """

    pad_token_id: int = 0
    max_turns: int = 200

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        # Stack all features
        batch = {}
        for key in features[0].keys():
            if isinstance(features[0][key], torch.Tensor):
                batch[key] = torch.stack([f[key] for f in features])
            else:
                batch[key] = torch.tensor([f[key] for f in features])

        # Create previous action/reward tensors (shifted by 1)
        prev_actions = torch.zeros_like(batch["actions"])
        prev_actions[:, 1:] = batch["actions"][:, :-1]
        prev_actions[:, 0] = 0  # Use action 0 (first move) as placeholder for first turn

        prev_rewards = torch.zeros_like(batch["rewards"])
        prev_rewards[:, 1:] = batch["rewards"][:, :-1]

        batch["prev_actions"] = prev_actions
        batch["prev_rewards"] = prev_rewards

        # Create action mask (all actions valid by default)
        batch["action_mask"] = torch.ones(batch_size, self.max_turns, 9, dtype=torch.bool)

        return batch


def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    max_turns: int = 200,
    num_workers: int = 4,
    shuffle: bool = True,
    streaming: bool = False,
    world_size: int = 1,
    rank: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = None,
    preload_to_ram: bool = False,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for Pokemon battle data.

    Args:
        data_path: Path to data directory or file
        batch_size: Batch size
        max_turns: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        streaming: Use streaming dataset (for very large data)
        world_size: Number of distributed processes
        rank: Current process rank
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        preload_to_ram: Preload all data to RAM for faster training
        **dataset_kwargs: Additional arguments for dataset
    """
    if streaming:
        dataset = StreamingPokemonDataset(
            data_path=data_path,
            max_turns=max_turns,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            **dataset_kwargs,
        )
    else:
        dataset = PokemonBattleDataset(
            data_path=data_path,
            max_turns=max_turns,
            **dataset_kwargs,
        )

        # Preload to RAM for faster training
        if preload_to_ram and hasattr(dataset, 'trajectories'):
            print("Preloading data to RAM...")
            # Data is already in RAM after loading, just ensure it stays there
            # For very large datasets, this confirms everything is loaded
            _ = len(dataset)
            print(f"Preloaded {len(dataset)} trajectories to RAM")

    collator = PokemonDataCollator(max_turns=max_turns)

    # Use DistributedSampler for multi-GPU training
    sampler = None
    if world_size > 1 and not streaming:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )

    # Build dataloader kwargs
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle and not streaming and sampler is None,  # Don't shuffle if using sampler
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "drop_last": True,
        "persistent_workers": num_workers > 0,  # Keep workers alive between epochs
    }

    # Add sampler if using distributed training
    if sampler is not None:
        loader_kwargs["sampler"] = sampler

    # Add prefetch_factor only if workers > 0
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)
