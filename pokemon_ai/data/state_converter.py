"""
Convert Metamon UniversalState dicts to tokenizer-friendly text observations.

The Metamon dataset stores states as complex nested dicts. We need to convert
these to a flat text format that our tokenizer can handle.

Output format (matching Metamon paper's 87-word observation):
<format> <choice_type> <player> <pokemon> <hp_pct> <status> <type1> <type2> ...
"""

from typing import Dict, Any, List, Optional


def normalize_name(name: str) -> str:
    """Normalize Pokemon/move/item names for tokenization"""
    if not name:
        return "unknown"
    # Remove spaces, hyphens, apostrophes, convert to lowercase
    return name.lower().replace(" ", "").replace("-", "").replace("'", "").replace(".", "")


def hp_bucket(hp_pct: float) -> str:
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


def get_pokemon_info(pokemon: Dict) -> List[str]:
    """Extract tokens from a pokemon dict"""
    tokens = []

    if not pokemon or not isinstance(pokemon, dict):
        return ["unknown", "hp100", "nostatus", "notype"]

    # Species name
    species = pokemon.get("species", pokemon.get("name", pokemon.get("ident", "unknown")))
    if isinstance(species, dict):
        species = species.get("name", "unknown")
    tokens.append(normalize_name(str(species)))

    # HP
    hp = pokemon.get("hp", pokemon.get("current_hp", 100))
    max_hp = pokemon.get("max_hp", pokemon.get("maxhp", 100))
    if max_hp and max_hp > 0:
        hp_pct = (hp / max_hp) * 100
    else:
        hp_pct = 100
    tokens.append(hp_bucket(hp_pct))

    # Status
    status = pokemon.get("status", pokemon.get("condition", ""))
    if isinstance(status, dict):
        status = status.get("id", "")
    status = str(status).lower() if status else "nostatus"
    if status in ["", "none", "null", "ok", "healthy"]:
        status = "nostatus"
    tokens.append(status)

    # Types
    types = pokemon.get("types", pokemon.get("type", []))
    if isinstance(types, str):
        types = [types]
    if not types:
        types = ["notype"]
    for t in types[:2]:
        tokens.append(normalize_name(str(t)))
    if len(types) < 2:
        tokens.append("notype")

    return tokens


def get_move_info(move: Dict) -> List[str]:
    """Extract tokens from a move dict"""
    tokens = []

    if not move or not isinstance(move, dict):
        return ["nomove", "notype", "physical"]

    # Move name
    name = move.get("name", move.get("id", move.get("move", "nomove")))
    tokens.append(normalize_name(str(name)))

    # Move type
    move_type = move.get("type", "notype")
    tokens.append(normalize_name(str(move_type)))

    # Category
    category = move.get("category", move.get("damage_class", "physical"))
    tokens.append(normalize_name(str(category)))

    return tokens


def convert_state_to_text(state: Dict[str, Any], format_id: str = "gen9ou") -> str:
    """
    Convert a Metamon UniversalState dict to a tokenizer-friendly text string.

    Args:
        state: The state dict from Metamon dataset
        format_id: Battle format (gen1ou, gen9ou, etc.)

    Returns:
        Space-separated string of tokens for the tokenizer
    """
    tokens = []

    # Format token
    tokens.append(f"<{normalize_name(format_id)}>")

    # Try to determine choice type from available actions
    if "force_switch" in state or state.get("forced_switch"):
        tokens.append("<forceswitch>")
    elif "force_move" in state or state.get("trapped"):
        tokens.append("<forcemove>")
    else:
        tokens.append("<anychoice>")

    # === PLAYER SECTION ===
    tokens.append("<player>")

    # Active Pokemon
    active = state.get("active", state.get("active_pokemon", state.get("my_active", {})))
    if isinstance(active, list):
        active = active[0] if active else {}
    tokens.extend(get_pokemon_info(active))

    # Available moves
    tokens.append("<moveset>")
    moves = state.get("moves", state.get("available_moves", state.get("legal_moves", [])))
    if isinstance(moves, dict):
        moves = list(moves.values())

    for move in moves[:4]:
        if isinstance(move, str):
            tokens.append(normalize_name(move))
        elif isinstance(move, dict):
            tokens.extend(get_move_info(move))

    # Pad to 4 moves
    while len([t for t in tokens if t not in ["<moveset>", "<player>"]]) < 12:
        tokens.append("nomove")

    # Team Pokemon (benched)
    team = state.get("team", state.get("pokemon", state.get("my_team", [])))
    if isinstance(team, dict):
        team = list(team.values())

    for i, poke in enumerate(team[:5]):  # Skip active, up to 5 benched
        if isinstance(poke, dict):
            tokens.extend(get_pokemon_info(poke)[:3])  # species, hp, status only

    # === OPPONENT SECTION ===
    tokens.append("<opponent>")

    # Opponent active
    opp_active = state.get("opponent_active", state.get("opp_active",
                 state.get("opponent", {}).get("active", {})))
    if isinstance(opp_active, list):
        opp_active = opp_active[0] if opp_active else {}
    tokens.extend(get_pokemon_info(opp_active))

    # Opponent team (what we've seen)
    opp_team = state.get("opponent_team", state.get("opp_team",
               state.get("opponent", {}).get("team", [])))
    if isinstance(opp_team, dict):
        opp_team = list(opp_team.values())

    for poke in opp_team[:5]:
        if isinstance(poke, dict):
            tokens.extend(get_pokemon_info(poke)[:3])

    # === CONDITIONS SECTION ===
    tokens.append("<conditions>")

    # Weather
    weather = state.get("weather", state.get("field", {}).get("weather", "noweather"))
    if isinstance(weather, dict):
        weather = weather.get("id", weather.get("type", "noweather"))
    weather = normalize_name(str(weather)) if weather else "noweather"
    if weather in ["none", "null", ""]:
        weather = "noweather"
    tokens.append(weather)

    # Side conditions (hazards)
    side_conditions = state.get("side_conditions", state.get("hazards", {}))
    if isinstance(side_conditions, dict):
        for condition in ["spikes", "stealthrock", "toxicspikes", "stickyweb",
                          "reflect", "lightscreen"]:
            if side_conditions.get(condition):
                tokens.append(condition)

    opp_side = state.get("opponent_side_conditions", state.get("opp_hazards", {}))
    if isinstance(opp_side, dict):
        for condition in ["spikes", "stealthrock", "toxicspikes", "stickyweb"]:
            if opp_side.get(condition):
                tokens.append(f"opp{condition}")

    # If no conditions, add placeholder
    if tokens[-1] == "<conditions>":
        tokens.append("noconditions")

    # Truncate to reasonable length (87 tokens like Metamon paper)
    tokens = tokens[:87]

    return " ".join(tokens)


def convert_metamon_state(state: Any, format_id: str = "gen9ou") -> str:
    """
    Main entry point for converting Metamon states.

    Handles various input types:
    - Dict: Normal Metamon UniversalState
    - String: Already converted, return as-is (unless it's a dict string)
    - Other: Return placeholder
    """
    if isinstance(state, str):
        # Check if it's a dict string representation
        if state.startswith("{") or state.startswith("{'"):
            # This is a dict that was converted to string incorrectly
            # Try to parse it back
            try:
                import ast
                state = ast.literal_eval(state)
            except (ValueError, SyntaxError):
                # Can't parse, return generic observation
                return f"<{normalize_name(format_id)}> <anychoice> <player> unknown hp100 nostatus notype <opponent> unknown hp100 nostatus notype <conditions> noweather"
        else:
            # Already a proper text observation
            return state

    if isinstance(state, dict):
        return convert_state_to_text(state, format_id)

    # Unknown type, return placeholder
    return f"<{normalize_name(format_id)}> <anychoice> <player> unknown hp100 nostatus notype <opponent> unknown hp100 nostatus notype <conditions> noweather"
