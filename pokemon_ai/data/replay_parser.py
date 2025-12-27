"""
Replay Parser - Download and process Pokemon Showdown replays

This module handles:
1. Downloading replays from Pokemon Showdown
2. Parsing replay logs into trajectory format
3. Reconstructing first-person POV from spectator data
4. Team inference for unrevealed information
"""

import os
import re
import json
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time

import numpy as np


@dataclass
class Pokemon:
    """Represents a Pokemon's state"""
    species: str = ""
    nickname: str = ""
    hp: float = 1.0
    max_hp: float = 100.0
    status: str = ""
    item: str = ""
    ability: str = ""
    moves: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    boosts: Dict[str, int] = field(default_factory=dict)
    revealed: bool = False


@dataclass
class BattleState:
    """Current state of a battle"""
    turn: int = 0
    weather: str = ""
    terrain: str = ""
    player_side_conditions: List[str] = field(default_factory=list)
    opponent_side_conditions: List[str] = field(default_factory=list)
    player_team: List[Pokemon] = field(default_factory=list)
    opponent_team: List[Pokemon] = field(default_factory=list)
    player_active: int = 0
    opponent_active: int = 0


class ReplayParser:
    """
    Parser for Pokemon Showdown replay logs.

    Converts spectator-view replays into first-person trajectory data
    suitable for training.
    """

    # Regex patterns for parsing
    TURN_PATTERN = re.compile(r'\|turn\|(\d+)')
    SWITCH_PATTERN = re.compile(r'\|switch\|p(\d)a: ([^|]+)\|([^|]+)\|(\d+)/(\d+)')
    MOVE_PATTERN = re.compile(r'\|move\|p(\d)a: ([^|]+)\|([^|]+)')
    DAMAGE_PATTERN = re.compile(r'\|-damage\|p(\d)a: ([^|]+)\|(\d+)/(\d+)')
    HEAL_PATTERN = re.compile(r'\|-heal\|p(\d)a: ([^|]+)\|(\d+)/(\d+)')
    FAINT_PATTERN = re.compile(r'\|faint\|p(\d)a: ([^|]+)')
    WIN_PATTERN = re.compile(r'\|win\|(.+)')
    WEATHER_PATTERN = re.compile(r'\|-weather\|(\w+)')
    STATUS_PATTERN = re.compile(r'\|-status\|p(\d)a: ([^|]+)\|(\w+)')

    MOVE_TO_IDX = {
        "move1": 0, "move2": 1, "move3": 2, "move4": 3,
        "switch1": 4, "switch2": 5, "switch3": 6, "switch4": 7, "switch5": 8,
    }

    def __init__(
        self,
        output_dir: str = "data/replays",
        formats: List[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats or ["gen1ou", "gen2ou", "gen3ou", "gen4ou"]

    async def download_replays(
        self,
        format_id: str,
        num_replays: int = 1000,
        min_rating: int = 1000,
    ) -> List[Dict]:
        """
        Download replays from Pokemon Showdown API.

        Args:
            format_id: Battle format (e.g., "gen1ou")
            num_replays: Number of replays to download
            min_rating: Minimum ELO rating filter
        """
        replays = []
        page = 1

        async with aiohttp.ClientSession() as session:
            while len(replays) < num_replays:
                url = f"https://replay.pokemonshowdown.com/search.json"
                params = {
                    "format": format_id,
                    "page": page,
                }

                try:
                    async with session.get(url, params=params) as response:
                        if response.status != 200:
                            break

                        data = await response.json()

                        if not data:
                            break

                        for replay_info in data:
                            rating = replay_info.get("rating") or 0
                            if rating >= min_rating:
                                replay_id = replay_info["id"]
                                replay_data = await self._fetch_replay(session, replay_id)
                                if replay_data:
                                    replays.append(replay_data)

                                    if len(replays) >= num_replays:
                                        break

                        page += 1
                        await asyncio.sleep(0.5)  # Rate limiting

                except Exception as e:
                    print(f"Error downloading replays: {e}")
                    break

        return replays

    async def _fetch_replay(self, session: aiohttp.ClientSession, replay_id: str) -> Optional[Dict]:
        """Fetch a single replay by ID"""
        url = f"https://replay.pokemonshowdown.com/{replay_id}.json"

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            print(f"Error fetching replay {replay_id}: {e}")

        return None

    def parse_replay(self, replay: Dict, pov_player: int = 1) -> Dict:
        """
        Parse a replay into trajectory format.

        Args:
            replay: Raw replay data from Pokemon Showdown
            pov_player: Which player's POV (1 or 2)

        Returns:
            Processed trajectory data
        """
        log = replay.get("log", "")
        format_id = replay.get("format", "").lower()
        rating = replay.get("rating", 0)
        players = replay.get("players", ["p1", "p2"])

        # Parse the log
        lines = log.split("\n")
        state = BattleState()
        trajectory = []
        current_turn_actions = {"p1": None, "p2": None}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Turn marker
            turn_match = self.TURN_PATTERN.match(line)
            if turn_match:
                # Save previous turn's state and actions
                if state.turn > 0:
                    obs = self._state_to_observation(state, pov_player)
                    action = current_turn_actions.get(f"p{pov_player}")
                    if action is not None:
                        trajectory.append({
                            "text_obs": obs["text"],
                            "numerical": obs["numerical"],
                            "action": action,
                            "reward": 0.0,
                            "done": False,
                        })

                state.turn = int(turn_match.group(1))
                current_turn_actions = {"p1": None, "p2": None}
                continue

            # Switch
            switch_match = self.SWITCH_PATTERN.match(line)
            if switch_match:
                player = int(switch_match.group(1))
                pokemon_name = switch_match.group(2)
                species = switch_match.group(3).split(",")[0]
                current_hp = int(switch_match.group(4))
                max_hp = int(switch_match.group(5))

                team = state.player_team if player == pov_player else state.opponent_team

                # Find or add pokemon
                pokemon_idx = self._find_pokemon(team, species, pokemon_name)
                if pokemon_idx == -1:
                    pokemon = Pokemon(
                        species=species,
                        nickname=pokemon_name,
                        hp=current_hp / max_hp,
                        max_hp=max_hp,
                        revealed=True,
                    )
                    team.append(pokemon)
                    pokemon_idx = len(team) - 1
                else:
                    team[pokemon_idx].hp = current_hp / max_hp
                    team[pokemon_idx].revealed = True

                if player == pov_player:
                    state.player_active = pokemon_idx
                    # Switch action: 4-8 for switching to pokemon 0-4 (excluding active)
                    # Clamp to valid range 4-8
                    switch_action = min(4 + pokemon_idx, 8)
                    current_turn_actions[f"p{player}"] = switch_action
                else:
                    state.opponent_active = pokemon_idx

                continue

            # Move
            move_match = self.MOVE_PATTERN.match(line)
            if move_match:
                player = int(move_match.group(1))
                move_name = move_match.group(3).lower().replace(" ", "")

                team = state.player_team if player == pov_player else state.opponent_team
                active = state.player_active if player == pov_player else state.opponent_active

                if active < len(team):
                    if move_name not in team[active].moves:
                        team[active].moves.append(move_name)

                    if player == pov_player:
                        move_idx = team[active].moves.index(move_name)
                        current_turn_actions[f"p{player}"] = min(move_idx, 3)

                continue

            # Damage
            damage_match = self.DAMAGE_PATTERN.match(line)
            if damage_match:
                player = int(damage_match.group(1))
                current_hp = int(damage_match.group(3))
                max_hp = int(damage_match.group(4))

                team = state.player_team if player == pov_player else state.opponent_team
                active = state.player_active if player == pov_player else state.opponent_active

                if active < len(team):
                    team[active].hp = current_hp / max_hp

                continue

            # Faint
            faint_match = self.FAINT_PATTERN.match(line)
            if faint_match:
                player = int(faint_match.group(1))

                team = state.player_team if player == pov_player else state.opponent_team
                active = state.player_active if player == pov_player else state.opponent_active

                if active < len(team):
                    team[active].hp = 0.0

                continue

            # Weather
            weather_match = self.WEATHER_PATTERN.match(line)
            if weather_match:
                state.weather = weather_match.group(1).lower()
                continue

            # Status
            status_match = self.STATUS_PATTERN.match(line)
            if status_match:
                player = int(status_match.group(1))
                status = status_match.group(3).lower()

                team = state.player_team if player == pov_player else state.opponent_team
                active = state.player_active if player == pov_player else state.opponent_active

                if active < len(team):
                    team[active].status = status

                continue

            # Win
            win_match = self.WIN_PATTERN.match(line)
            if win_match:
                winner = win_match.group(1)
                won = (winner == players[pov_player - 1])

                # Add final observation with win/loss reward
                if trajectory:
                    trajectory[-1]["done"] = True
                    trajectory[-1]["reward"] = 100.0 if won else -100.0

        # Compute intermediate rewards
        trajectory = self._compute_rewards(trajectory)

        return {
            "format": format_id,
            "rating": rating,
            "won": trajectory[-1].get("reward", 0) > 0 if trajectory else False,
            "steps": trajectory,
        }

    def _find_pokemon(self, team: List[Pokemon], species: str, nickname: str) -> int:
        """Find pokemon in team by species or nickname"""
        for i, pokemon in enumerate(team):
            if pokemon.species.lower() == species.lower():
                return i
            if pokemon.nickname.lower() == nickname.lower():
                return i
        return -1

    def _state_to_observation(self, state: BattleState, pov_player: int) -> Dict:
        """Convert battle state to observation format"""
        text_parts = []

        # Format
        text_parts.append("<gen1ou>")  # Default format, should be passed in

        # Choice type
        text_parts.append("<anychoice>")

        # Player info
        text_parts.append("<player>")
        if state.player_team and state.player_active < len(state.player_team):
            active = state.player_team[state.player_active]
            text_parts.extend([
                active.species.lower(),
                active.item.lower() if active.item else "unknownitem",
                active.ability.lower() if active.ability else "unknownability",
            ])
            text_parts.extend(active.types[:2] if active.types else ["notype", "notype"])
            text_parts.append(active.status if active.status else "nostatus")

            # Moves
            for i in range(4):
                if i < len(active.moves):
                    text_parts.append("<move>")
                    text_parts.append(active.moves[i])
                    text_parts.append("normal")  # Type placeholder
                    text_parts.append("physical")  # Category placeholder
                else:
                    text_parts.extend(["<move>", "<blank>", "<blank>", "<blank>"])

            # Team switches
            for i, pokemon in enumerate(state.player_team):
                if i != state.player_active:
                    text_parts.append("<switch>")
                    text_parts.append(pokemon.species.lower())
                    text_parts.append(pokemon.item.lower() if pokemon.item else "unknownitem")
                    text_parts.append(pokemon.ability.lower() if pokemon.ability else "unknownability")

        # Opponent info
        text_parts.append("<opponent>")
        if state.opponent_team and state.opponent_active < len(state.opponent_team):
            active = state.opponent_team[state.opponent_active]
            text_parts.extend([
                active.species.lower(),
                "unknownitem",
                "unknownability",
            ])
            text_parts.extend(active.types[:2] if active.types else ["notype", "notype"])
            text_parts.append(active.status if active.status else "nostatus")

        # Conditions
        text_parts.append("<conditions>")
        text_parts.append(state.weather if state.weather else "noweather")
        text_parts.append("noconditions")
        text_parts.append("noconditions")

        # Previous moves
        text_parts.extend(["<player_prev>", "nomove"])
        text_parts.extend(["<opp_prev>", "nomove"])

        # Numerical features
        numerical = []
        # HP values
        for pokemon in state.player_team[:6]:
            numerical.append(pokemon.hp)
        while len(numerical) < 6:
            numerical.append(0.0)

        for pokemon in state.opponent_team[:6]:
            numerical.append(pokemon.hp)
        while len(numerical) < 12:
            numerical.append(0.0)

        # Pad to 48 features
        while len(numerical) < 48:
            numerical.append(0.0)

        return {
            "text": " ".join(text_parts),
            "numerical": numerical[:48],
        }

    def _compute_rewards(self, trajectory: List[Dict]) -> List[Dict]:
        """Compute intermediate rewards based on damage/faints"""
        for i, step in enumerate(trajectory):
            if step.get("reward", 0) == 0:
                step["reward"] = 0.0

        return trajectory

    def save_trajectories(self, trajectories: List[Dict], filename: str):
        """Save processed trajectories to file"""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump({"trajectories": trajectories}, f)

        print(f"Saved {len(trajectories)} trajectories to {output_path}")


async def download_and_process_replays(
    output_dir: str = "data/replays",
    formats: List[str] = None,
    num_replays_per_format: int = 10000,
    min_rating: int = 1200,
):
    """
    Main function to download and process replays.

    This creates the training dataset from Pokemon Showdown replays.
    """
    formats = formats or ["gen1ou", "gen2ou", "gen3ou", "gen4ou"]
    parser = ReplayParser(output_dir=output_dir, formats=formats)

    all_trajectories = []

    for format_id in formats:
        print(f"Downloading {format_id} replays...")
        replays = await parser.download_replays(
            format_id=format_id,
            num_replays=num_replays_per_format,
            min_rating=min_rating,
        )

        print(f"Processing {len(replays)} replays...")
        for replay in replays:
            # Process from both players' perspectives
            for pov in [1, 2]:
                try:
                    traj = parser.parse_replay(replay, pov_player=pov)
                    if traj["steps"]:
                        all_trajectories.append(traj)
                except Exception as e:
                    print(f"Error processing replay: {e}")
                    continue

        # Save periodically
        parser.save_trajectories(all_trajectories, f"{format_id}_trajectories.json")

    print(f"Total trajectories: {len(all_trajectories)}")
    return all_trajectories


if __name__ == "__main__":
    asyncio.run(download_and_process_replays())
