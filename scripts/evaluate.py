#!/usr/bin/env python3
"""
Evaluate trained model against baselines.

This script runs battles against various opponents to measure win rate.

Usage:
    # vs Random bot (sanity check - should win easily)
    python scripts/evaluate.py --checkpoint checkpoints/metamon_shards/final --opponent random --num_battles 100

    # vs Max Power bot (uses highest base power move)
    python scripts/evaluate.py --checkpoint checkpoints/metamon_shards/final --opponent max_power --num_battles 100

    # vs Another trained model
    python scripts/evaluate.py --checkpoint checkpoints/v1/final --opponent checkpoints/v2/final --num_battles 100

Requirements:
    pip install poke-env
    # Also need a local Pokemon Showdown server for offline battles
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from poke_env.player import RandomPlayer, MaxBasePowerPlayer
    from poke_env import PlayerConfiguration, ServerConfiguration
    HAS_POKE_ENV = True
except ImportError:
    print("ERROR: poke-env not installed. Run: pip install poke-env")
    HAS_POKE_ENV = False

from pokemon_ai.agents import create_agent


@dataclass
class EvalResults:
    """Evaluation results"""
    wins: int = 0
    losses: int = 0
    total: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total, 1)

    def __str__(self) -> str:
        return f"Win Rate: {self.win_rate:.1%} ({self.wins}/{self.total})"


async def evaluate_vs_random(
    agent,
    num_battles: int = 100,
    battle_format: str = "gen9ou",
) -> EvalResults:
    """Evaluate against random player"""
    opponent = RandomPlayer(battle_format=battle_format)

    await agent.battle_against(opponent, n_battles=num_battles)

    results = EvalResults(
        wins=agent.n_won_battles,
        losses=agent.n_lost_battles,
        total=num_battles,
    )
    return results


async def evaluate_vs_max_power(
    agent,
    num_battles: int = 100,
    battle_format: str = "gen9ou",
) -> EvalResults:
    """Evaluate against max base power player"""
    opponent = MaxBasePowerPlayer(battle_format=battle_format)

    await agent.battle_against(opponent, n_battles=num_battles)

    results = EvalResults(
        wins=agent.n_won_battles,
        losses=agent.n_lost_battles,
        total=num_battles,
    )
    return results


async def evaluate_vs_model(
    agent,
    opponent_checkpoint: str,
    num_battles: int = 100,
    battle_format: str = "gen9ou",
    model_size: str = "base",
) -> EvalResults:
    """Evaluate against another trained model"""
    opponent = create_agent(
        checkpoint_path=opponent_checkpoint,
        model_size=model_size,
        battle_format=battle_format,
    )

    await agent.battle_against(opponent, n_battles=num_battles)

    results = EvalResults(
        wins=agent.n_won_battles,
        losses=agent.n_lost_battles,
        total=num_battles,
    )
    return results


async def main_async(args):
    """Main evaluation loop"""
    print("=" * 60)
    print("POKEMON AI EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Opponent: {args.opponent}")
    print(f"Format: {args.format}")
    print(f"Battles: {args.num_battles}")
    print("=" * 60)

    # Create agent
    agent = create_agent(
        checkpoint_path=args.checkpoint,
        model_size=args.model_size,
        battle_format=args.format,
        deterministic=args.deterministic,
    )

    # Run evaluation
    if args.opponent == "random":
        print("\nEvaluating vs Random Player...")
        results = await evaluate_vs_random(
            agent, args.num_battles, args.format
        )
    elif args.opponent == "max_power":
        print("\nEvaluating vs Max Power Player...")
        results = await evaluate_vs_max_power(
            agent, args.num_battles, args.format
        )
    else:
        # Assume it's a checkpoint path
        print(f"\nEvaluating vs model: {args.opponent}")
        results = await evaluate_vs_model(
            agent, args.opponent, args.num_battles, args.format, args.model_size
        )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(results)
    print("=" * 60)

    # Performance benchmarks
    print("\nBenchmarks for reference:")
    print("  vs Random: Should be >90% win rate")
    print("  vs Max Power: Should be >70% win rate")
    print("  vs Heuristic Bots: Target >50% for competent play")
    print("  vs Ladder 1500: Target >50% for human-level")
    print("  vs Ladder 1800+: Target for superhuman")

    return results


def main():
    if not HAS_POKE_ENV:
        print("\nTo run evaluation, install poke-env:")
        print("  pip install poke-env")
        print("\nYou'll also need a local Pokemon Showdown server.")
        print("See: https://github.com/hsahovic/poke-env#pokemon-showdown-server")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Evaluate Pokemon AI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        help="Opponent: 'random', 'max_power', or checkpoint path",
    )
    parser.add_argument(
        "--num_battles",
        type=int,
        default=100,
        help="Number of battles to run",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen9ou",
        help="Battle format",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        help="Model size (small, base, large, xl)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic (argmax) action selection",
    )

    args = parser.parse_args()

    # Run async evaluation
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
