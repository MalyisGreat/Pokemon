"""
Parser for PokeChamp/Pokemon Showdown battle log format.

The format is pipe-delimited with lines like:
|move|p1a: pikachu|thunderbolt|p2a: spiritomb
|-damage|p2a: spiritomb|61/100
|switch|p1a: lapras|lapras, f|100/100

We extract turn-by-turn states and actions for RL training.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Pokemon:
    """Represents a Pokemon's current state"""
    species: str = "unknown"
    hp_pct: float = 100.0
    status: str = ""
    is_active: bool = False


@dataclass
class BattleState:
    """State of battle at a given turn"""
    turn: int = 0
    player_active: Pokemon = field(default_factory=Pokemon)
    player_team: List[Pokemon] = field(default_factory=list)
    opponent_active: Pokemon = field(default_factory=Pokemon)
    opponent_team: List[Pokemon] = field(default_factory=list)
    weather: str = ""
    player_side_conditions: List[str] = field(default_factory=list)
    opponent_side_conditions: List[str] = field(default_factory=list)


@dataclass
class TurnAction:
    """An action taken during a turn"""
    action_type: str  # "move" or "switch"
    action_name: str  # move name or pokemon name
    player: str  # "p1" or "p2"


def parse_hp(hp_str: str) -> float:
    """Parse HP string like '61/100' or '94/100 tox' to percentage"""
    if not hp_str:
        return 100.0
    # Remove status conditions
    hp_str = hp_str.split()[0]
    if hp_str == "0" or hp_str == "0 fnt":
        return 0.0
    if "/" in hp_str:
        parts = hp_str.split("/")
        try:
            current = float(parts[0])
            max_hp = float(parts[1])
            return (current / max_hp) * 100 if max_hp > 0 else 0.0
        except ValueError:
            return 100.0
    # Already a percentage or raw value
    try:
        return float(hp_str)
    except ValueError:
        return 100.0


def parse_pokemon_name(pokemon_str: str) -> str:
    """Parse pokemon name from strings like 'p1a: pikachu' or 'pikachu, f'"""
    # Remove player prefix
    if ": " in pokemon_str:
        pokemon_str = pokemon_str.split(": ")[1]
    # Remove gender
    if ", " in pokemon_str:
        pokemon_str = pokemon_str.split(", ")[0]
    return pokemon_str.lower().strip()


def parse_status(line: str) -> Tuple[str, str]:
    """Parse status from |-status| line, returns (pokemon, status)"""
    parts = line.split("|")
    if len(parts) >= 4:
        return parse_pokemon_name(parts[2]), parts[3]
    return "", ""


class PokechampBattleParser:
    """
    Parses a PokeChamp battle log into turn-by-turn states and actions.

    We focus on one player's perspective (the winner by default).
    """

    # Map move names to action indices (0-3 = moves, 4-8 = switches)
    # For simplicity, we'll use a hash-based approach

    def __init__(self, perspective: str = "winner"):
        """
        Args:
            perspective: "winner", "loser", "p1", or "p2" - whose perspective to take
        """
        self.perspective = perspective

    def parse(self, battle_log: str) -> List[Dict[str, Any]]:
        """
        Parse a battle log into a list of (state, action) pairs.

        Returns list of dicts with:
        - text_obs: text observation string
        - action: action index 0-8
        - reward: 0.0 for most turns, 1.0 for win, -1.0 for loss
        - done: True for final turn
        """
        lines = battle_log.strip().split("\n")

        # First pass: determine format, players, and winner
        game_info = self._parse_game_info(lines)
        if not game_info:
            return []

        format_id = game_info.get("format", "gen9ou")
        winner = game_info.get("winner", "")
        p1_name = game_info.get("p1", "")
        p2_name = game_info.get("p2", "")

        # Determine which player we're following
        if self.perspective == "winner":
            our_player = "p1" if winner == p1_name else "p2"
        elif self.perspective == "loser":
            our_player = "p2" if winner == p1_name else "p1"
        else:
            our_player = self.perspective

        opp_player = "p2" if our_player == "p1" else "p1"
        we_won = (our_player == "p1" and winner == p1_name) or \
                 (our_player == "p2" and winner == p2_name)

        # Second pass: extract turn-by-turn state and actions
        turns = self._parse_turns(lines, our_player, opp_player, format_id)

        # Add rewards
        for i, turn in enumerate(turns):
            turn["reward"] = 0.0
            turn["done"] = False

        if turns:
            turns[-1]["done"] = True
            turns[-1]["reward"] = 1.0 if we_won else -1.0

        return turns

    def _parse_game_info(self, lines: List[str]) -> Dict[str, str]:
        """Extract game metadata"""
        info = {}

        for line in lines:
            if not line.startswith("|"):
                continue
            parts = line.split("|")

            if len(parts) >= 3:
                cmd = parts[1]

                if cmd == "player" and len(parts) >= 4:
                    # |player|p1|username|avatar|elo
                    player_id = parts[2]
                    username = parts[3]
                    info[player_id] = username

                elif cmd == "tier" or cmd == "gen":
                    # |tier|[gen 6] ou or |gen|6
                    tier = parts[2].lower()
                    # Extract format like "gen6ou"
                    match = re.search(r'gen\s*(\d+)\]?\s*(\w+)?', tier)
                    if match:
                        gen = match.group(1)
                        tier_name = match.group(2) or "ou"
                        info["format"] = f"gen{gen}{tier_name}"

                elif cmd == "win" and len(parts) >= 3:
                    info["winner"] = parts[2]

        return info

    def _parse_turns(self, lines: List[str], our_player: str, opp_player: str,
                     format_id: str) -> List[Dict[str, Any]]:
        """Parse turn-by-turn states and actions"""
        turns = []
        current_turn = 0

        # Track Pokemon state
        player_pokemon: Dict[str, Pokemon] = {}
        opponent_pokemon: Dict[str, Pokemon] = {}
        player_active = ""
        opponent_active = ""
        weather = ""
        player_hazards = []
        opponent_hazards = []

        # Track actions for current turn
        current_action = None

        for line in lines:
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue

            cmd = parts[1]

            # Turn marker
            if cmd == "turn" and len(parts) >= 3:
                # Save previous turn if we have an action
                if current_turn > 0 and current_action is not None:
                    state_text = self._build_state_text(
                        format_id, player_active, player_pokemon,
                        opponent_active, opponent_pokemon,
                        weather, player_hazards, opponent_hazards
                    )
                    turns.append({
                        "text_obs": state_text,
                        "action": current_action,
                    })

                current_turn = int(parts[2])
                current_action = None

            # Switch
            elif cmd == "switch" or cmd == "drag":
                if len(parts) >= 5:
                    pokemon_slot = parts[2]  # e.g., "p1a: pikachu"
                    species = parse_pokemon_name(parts[3])
                    hp_pct = parse_hp(parts[4])

                    player = "p1" if pokemon_slot.startswith("p1") else "p2"

                    pokemon = Pokemon(species=species, hp_pct=hp_pct, is_active=True)

                    if player == our_player:
                        player_pokemon[species] = pokemon
                        player_active = species
                        # Switch action (indices 4-8)
                        if current_action is None:
                            current_action = self._switch_to_action_idx(species, player_pokemon)
                    else:
                        opponent_pokemon[species] = pokemon
                        opponent_active = species

            # Move
            elif cmd == "move":
                if len(parts) >= 4:
                    pokemon_slot = parts[2]
                    move_name = parts[3].lower()

                    player = "p1" if pokemon_slot.startswith("p1") else "p2"

                    if player == our_player and current_action is None:
                        current_action = self._move_to_action_idx(move_name)

            # Damage
            elif cmd == "-damage" or cmd == "-heal":
                if len(parts) >= 4:
                    pokemon_slot = parts[2]
                    hp_pct = parse_hp(parts[3])
                    species = parse_pokemon_name(pokemon_slot)

                    player = "p1" if pokemon_slot.startswith("p1") else "p2"

                    if player == our_player:
                        if species in player_pokemon:
                            player_pokemon[species].hp_pct = hp_pct
                    else:
                        if species in opponent_pokemon:
                            opponent_pokemon[species].hp_pct = hp_pct

            # Status
            elif cmd == "-status":
                if len(parts) >= 4:
                    pokemon_slot = parts[2]
                    status = parts[3]
                    species = parse_pokemon_name(pokemon_slot)

                    player = "p1" if pokemon_slot.startswith("p1") else "p2"

                    if player == our_player:
                        if species in player_pokemon:
                            player_pokemon[species].status = status
                    else:
                        if species in opponent_pokemon:
                            opponent_pokemon[species].status = status

            # Cure status
            elif cmd == "-curestatus":
                if len(parts) >= 3:
                    pokemon_slot = parts[2]
                    species = parse_pokemon_name(pokemon_slot)

                    player = "p1" if pokemon_slot.startswith("p1") else "p2"

                    if player == our_player:
                        if species in player_pokemon:
                            player_pokemon[species].status = ""
                    else:
                        if species in opponent_pokemon:
                            opponent_pokemon[species].status = ""

            # Weather
            elif cmd == "-weather":
                if len(parts) >= 3:
                    weather = parts[2].lower()
                    if weather == "none":
                        weather = ""

            # Side conditions (hazards)
            elif cmd == "-sidestart":
                if len(parts) >= 4:
                    side = parts[2]  # "p1: username" or "p2: username"
                    condition = parts[3].lower()

                    # Normalize condition name
                    condition = condition.replace("move: ", "").replace(" ", "")

                    if side.startswith("p1"):
                        if our_player == "p1":
                            if condition not in player_hazards:
                                player_hazards.append(condition)
                        else:
                            if condition not in opponent_hazards:
                                opponent_hazards.append(condition)
                    else:
                        if our_player == "p2":
                            if condition not in player_hazards:
                                player_hazards.append(condition)
                        else:
                            if condition not in opponent_hazards:
                                opponent_hazards.append(condition)

            elif cmd == "-sideend":
                if len(parts) >= 4:
                    side = parts[2]
                    condition = parts[3].lower().replace("move: ", "").replace(" ", "")

                    if side.startswith("p1"):
                        target = player_hazards if our_player == "p1" else opponent_hazards
                    else:
                        target = player_hazards if our_player == "p2" else opponent_hazards

                    if condition in target:
                        target.remove(condition)

            # Faint
            elif cmd == "faint":
                if len(parts) >= 3:
                    pokemon_slot = parts[2]
                    species = parse_pokemon_name(pokemon_slot)

                    player = "p1" if pokemon_slot.startswith("p1") else "p2"

                    if player == our_player:
                        if species in player_pokemon:
                            player_pokemon[species].hp_pct = 0.0
                    else:
                        if species in opponent_pokemon:
                            opponent_pokemon[species].hp_pct = 0.0

        # Save last turn
        if current_turn > 0 and current_action is not None:
            state_text = self._build_state_text(
                format_id, player_active, player_pokemon,
                opponent_active, opponent_pokemon,
                weather, player_hazards, opponent_hazards
            )
            turns.append({
                "text_obs": state_text,
                "action": current_action,
            })

        return turns

    def _build_state_text(self, format_id: str, player_active: str,
                          player_pokemon: Dict[str, Pokemon],
                          opponent_active: str, opponent_pokemon: Dict[str, Pokemon],
                          weather: str, player_hazards: List[str],
                          opponent_hazards: List[str]) -> str:
        """Build text observation from state"""
        tokens = []

        # Format
        tokens.append(f"<{format_id}>")
        tokens.append("<anychoice>")

        # Player section
        tokens.append("<player>")
        if player_active and player_active in player_pokemon:
            p = player_pokemon[player_active]
            tokens.append(p.species)
            tokens.append(self._hp_bucket(p.hp_pct))
            tokens.append(p.status if p.status else "nostatus")
        else:
            tokens.extend(["unknown", "hp100", "nostatus"])

        # Team (benched)
        for species, pokemon in player_pokemon.items():
            if species != player_active:
                tokens.append(species)
                tokens.append(self._hp_bucket(pokemon.hp_pct))
                tokens.append(pokemon.status if pokemon.status else "nostatus")

        # Opponent section
        tokens.append("<opponent>")
        if opponent_active and opponent_active in opponent_pokemon:
            p = opponent_pokemon[opponent_active]
            tokens.append(p.species)
            tokens.append(self._hp_bucket(p.hp_pct))
            tokens.append(p.status if p.status else "nostatus")
        else:
            tokens.extend(["unknown", "hp100", "nostatus"])

        # Opponent team (what we've seen)
        for species, pokemon in opponent_pokemon.items():
            if species != opponent_active:
                tokens.append(species)
                tokens.append(self._hp_bucket(pokemon.hp_pct))
                tokens.append(pokemon.status if pokemon.status else "nostatus")

        # Conditions
        tokens.append("<conditions>")
        if weather:
            tokens.append(weather)
        for hazard in player_hazards:
            tokens.append(hazard)
        for hazard in opponent_hazards:
            tokens.append(f"opp{hazard}")
        if not weather and not player_hazards and not opponent_hazards:
            tokens.append("noweather")

        return " ".join(tokens[:87])  # Limit to 87 tokens

    def _hp_bucket(self, hp_pct: float) -> str:
        """Convert HP percentage to bucket token"""
        if hp_pct <= 0:
            return "hp0"
        elif hp_pct <= 25:
            return "hp25"
        elif hp_pct <= 50:
            return "hp50"
        elif hp_pct <= 75:
            return "hp75"
        else:
            return "hp100"

    def _move_to_action_idx(self, move_name: str) -> int:
        """Convert move name to action index 0-3"""
        # Simple hash to get consistent index
        return hash(move_name) % 4

    def _switch_to_action_idx(self, species: str, team: Dict[str, Pokemon]) -> int:
        """Convert switch to action index 4-8"""
        # Get position in team (excluding fainted)
        alive_team = [s for s, p in team.items() if p.hp_pct > 0]
        try:
            idx = alive_team.index(species)
            return 4 + min(idx, 4)  # Cap at index 8
        except ValueError:
            return 4  # Default switch
