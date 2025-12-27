#!/usr/bin/env python3
"""
Self-Play Data Generation

This is Phase 2 of training - after offline RL on human data.

The model plays against itself (or past versions) to generate
unlimited high-quality training data.

How it works:
1. Load trained model from Phase 1 (offline RL)
2. Run games: model vs model (or model vs past checkpoint)
3. Save trajectories to disk
4. These become training data for next iteration

Usage:
    python scripts/self_play.py --checkpoint checkpoints/h100_base/final --num_games 10000
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


@dataclass
class SelfPlayConfig:
    """Configuration for self-play"""
    checkpoint_path: str
    output_dir: str = "data/self_play"
    num_games: int = 10000
    games_per_batch: int = 100

    # Opponent selection
    use_past_checkpoints: bool = True  # Play against older versions
    past_checkpoint_prob: float = 0.3  # 30% games vs past versions

    # Game settings
    format: str = "gen9ou"
    temperature: float = 1.0  # Sampling temperature

    # Hardware
    device: str = "cuda"
    num_parallel_games: int = 32  # Batch games for efficiency


@dataclass
class GameTrajectory:
    """A single game's trajectory"""
    format: str
    player1_trajectory: List[Dict] = field(default_factory=list)
    player2_trajectory: List[Dict] = field(default_factory=list)
    winner: int = 0  # 1 or 2
    num_turns: int = 0

    def to_training_format(self) -> List[Dict]:
        """Convert to training trajectories (one per player)"""
        trajectories = []

        # Player 1's perspective
        if self.player1_trajectory:
            p1_traj = {
                "format": self.format,
                "rating": 1600,  # Self-play rating placeholder
                "won": self.winner == 1,
                "steps": self.player1_trajectory,
            }
            trajectories.append(p1_traj)

        # Player 2's perspective
        if self.player2_trajectory:
            p2_traj = {
                "format": self.format,
                "rating": 1600,
                "won": self.winner == 2,
                "steps": self.player2_trajectory,
            }
            trajectories.append(p2_traj)

        return trajectories


class SelfPlayGenerator:
    """Generate training data through self-play"""

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.past_models = []
        self.games_generated = 0
        self.trajectories_saved = 0

    def load_model(self):
        """Load the trained model"""
        from pokemon_ai.models import create_pokemon_transformer
        from safetensors.torch import load_file

        checkpoint_path = Path(self.config.checkpoint_path)

        # Load config
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
            size = model_config.get("size", "base")
        else:
            size = "base"

        # Create model
        self.model = create_pokemon_transformer(
            size=size,
            use_flash_attention=torch.cuda.is_available(),
        )

        # Load weights
        weights_path = checkpoint_path / "model.safetensors"
        if weights_path.exists():
            state_dict = load_file(weights_path)
            self.model.load_state_dict(state_dict)
        else:
            # Try pytorch format
            weights_path = checkpoint_path / "pytorch_model.bin"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"No model weights found in {checkpoint_path}")

        self.model = self.model.to(self.config.device)
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")

    def load_past_checkpoints(self, checkpoint_dir: Path):
        """Load past model versions for diverse opponents"""
        # Find all checkpoints
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))

        if not checkpoints:
            print("No past checkpoints found, using only current model")
            return

        # Load a sample of past checkpoints (not all, to save memory)
        sample_size = min(5, len(checkpoints))
        sampled = random.sample(checkpoints, sample_size)

        print(f"Loading {len(sampled)} past checkpoints for opponent pool")

        for ckpt in sampled:
            try:
                from pokemon_ai.models import create_pokemon_transformer
                from safetensors.torch import load_file

                model = create_pokemon_transformer(size="base")
                weights_path = ckpt / "model.safetensors"
                if weights_path.exists():
                    state_dict = load_file(weights_path)
                    model.load_state_dict(state_dict)
                    model = model.to(self.config.device)
                    model.eval()
                    self.past_models.append(model)
                    print(f"  Loaded {ckpt.name}")
            except Exception as e:
                print(f"  Failed to load {ckpt.name}: {e}")

    def select_opponent(self):
        """Select opponent model for a game"""
        if self.past_models and random.random() < self.config.past_checkpoint_prob:
            return random.choice(self.past_models)
        return self.model

    def run_game(self, player1_model, player2_model) -> GameTrajectory:
        """
        Run a single game between two models.

        NOTE: This is a placeholder - real implementation needs:
        1. Pokemon Showdown simulator integration (poke-env)
        2. State encoding/decoding
        3. Action sampling from model outputs
        """
        # Placeholder implementation
        trajectory = GameTrajectory(format=self.config.format)

        # TODO: Implement actual game loop with poke-env
        # For now, return empty trajectory
        # Real implementation:
        #
        # from poke_env.player import Player
        #
        # class ModelPlayer(Player):
        #     def __init__(self, model, ...):
        #         self.model = model
        #
        #     def choose_move(self, battle):
        #         obs = encode_battle_state(battle)
        #         with torch.no_grad():
        #             logits = self.model(obs)
        #             action = sample_action(logits, temperature)
        #         return decode_action(action, battle)
        #
        # player1 = ModelPlayer(player1_model)
        # player2 = ModelPlayer(player2_model)
        # await player1.battle_against(player2, n_battles=1)

        trajectory.num_turns = random.randint(10, 50)  # Placeholder
        trajectory.winner = random.choice([1, 2])  # Placeholder

        return trajectory

    async def generate_games(self, num_games: int) -> List[GameTrajectory]:
        """Generate multiple games"""
        trajectories = []

        for i in range(num_games):
            opponent = self.select_opponent()

            # Randomly assign sides
            if random.random() < 0.5:
                game = self.run_game(self.model, opponent)
            else:
                game = self.run_game(opponent, self.model)

            trajectories.append(game)
            self.games_generated += 1

        return trajectories

    def save_trajectories(self, trajectories: List[GameTrajectory], batch_num: int):
        """Save trajectories to disk"""
        training_data = []

        for game in trajectories:
            training_data.extend(game.to_training_format())

        if training_data:
            filename = self.output_dir / f"self_play_batch_{batch_num:06d}.json"
            with open(filename, "w") as f:
                json.dump({"trajectories": training_data}, f)

            self.trajectories_saved += len(training_data)
            print(f"Saved batch {batch_num}: {len(training_data)} trajectories")

    async def run(self):
        """Main self-play loop"""
        print("=" * 60)
        print("SELF-PLAY DATA GENERATION")
        print("=" * 60)
        print(f"Target games: {self.config.num_games}")
        print(f"Output: {self.output_dir}")
        print("=" * 60)

        # Load models
        print("\nLoading model...")
        self.load_model()

        if self.config.use_past_checkpoints:
            checkpoint_dir = Path(self.config.checkpoint_path).parent
            self.load_past_checkpoints(checkpoint_dir)

        # Generate games in batches
        print("\nStarting self-play...")
        start_time = time.time()
        batch_num = 0

        while self.games_generated < self.config.num_games:
            batch_size = min(
                self.config.games_per_batch,
                self.config.num_games - self.games_generated
            )

            trajectories = await self.generate_games(batch_size)
            self.save_trajectories(trajectories, batch_num)
            batch_num += 1

            # Progress
            elapsed = time.time() - start_time
            rate = self.games_generated / elapsed if elapsed > 0 else 0
            print(
                f"Progress: {self.games_generated}/{self.config.num_games} games "
                f"({rate:.1f} games/s)"
            )

        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("SELF-PLAY COMPLETE")
        print("=" * 60)
        print(f"Games generated: {self.games_generated}")
        print(f"Trajectories saved: {self.trajectories_saved}")
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Rate: {self.games_generated/elapsed:.1f} games/s")
        print(f"Output: {self.output_dir}")
        print("\nNext steps:")
        print("1. Combine with human data")
        print("2. Retrain model on mixed dataset")
        print("3. Repeat self-play with improved model")


def main():
    parser = argparse.ArgumentParser(description="Generate self-play training data")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        default="data/self_play",
        help="Output directory for trajectories",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=10000,
        help="Number of games to generate",
    )
    parser.add_argument(
        "--format",
        default="gen9ou",
        help="Battle format",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    config = SelfPlayConfig(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        num_games=args.num_games,
        format=args.format,
        temperature=args.temperature,
    )

    generator = SelfPlayGenerator(config)
    asyncio.run(generator.run())


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NOTE: Full self-play requires poke-env integration")
    print("This script shows the structure - game simulation is placeholder")
    print("=" * 60)
    print("\nTo use real battles, install: pip install poke-env")
    print("And implement the battle loop in run_game()")
    print("=" * 60 + "\n")

    main()
