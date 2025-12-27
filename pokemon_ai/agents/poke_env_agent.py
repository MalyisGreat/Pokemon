"""
Poke-env Agent Wrapper

Wraps our trained PokemonTransformer model to play battles via poke-env.
This allows evaluation against bots and on the Pokemon Showdown ladder.

Usage:
    from pokemon_ai.agents import create_agent
    agent = create_agent("checkpoints/metamon_shards/final")
    # Agent can now battle via poke-env
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer
    from poke_env.environment import Battle, Pokemon, Move
    HAS_POKE_ENV = True
except ImportError:
    HAS_POKE_ENV = False
    Player = object  # Placeholder for type hints

from pokemon_ai.models import PokemonTransformer, create_pokemon_transformer
from pokemon_ai.data.tokenizer import PokemonTokenizer


class PokemonAIAgent(Player if HAS_POKE_ENV else object):
    """
    A poke-env player that uses our trained transformer model.

    The agent converts battle states to our observation format,
    runs inference, and selects actions.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "base",
        device: str = "cuda",
        temperature: float = 1.0,
        deterministic: bool = False,
        battle_format: str = "gen9ou",
        **kwargs
    ):
        if not HAS_POKE_ENV:
            raise ImportError("poke-env not installed. Run: pip install poke-env")

        super().__init__(battle_format=battle_format, **kwargs)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.deterministic = deterministic
        self.tokenizer = PokemonTokenizer()

        # Load model
        self.model = self._load_model(checkpoint_path, model_size)
        self.model.eval()

        # Track battle history for sequential decisions
        self.battle_history: Dict[str, List[Dict]] = {}

        # Action space: 0-3 moves, 4-8 switches, 9-12 tera+moves
        self.num_actions = 13

    def _load_model(self, checkpoint_path: str, model_size: str) -> PokemonTransformer:
        """Load trained model from checkpoint"""
        checkpoint_dir = Path(checkpoint_path)

        # Try to load model
        model_path = checkpoint_dir / "model.pt"
        if not model_path.exists():
            # Try DeepSpeed checkpoint format
            model_path = checkpoint_dir / "mp_rank_00_model_states.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"No model found in {checkpoint_dir}")

        # Create model architecture
        model = create_pokemon_transformer(
            size=model_size,
            use_flash_attention=False,  # Inference doesn't need flash
            use_gradient_checkpointing=False,
        )

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)

        print(f"Loaded model from {checkpoint_path}")
        return model

    def _battle_to_observation(self, battle: Battle) -> Dict[str, torch.Tensor]:
        """Convert poke-env battle state to our observation format"""
        tokens = []

        # Format token
        format_str = battle.battle_tag.split("-")[0] if battle.battle_tag else "gen9ou"
        tokens.append(f"<{format_str}>")

        # Choice type
        if battle.force_switch:
            tokens.append("<forceswitch>")
        else:
            tokens.append("<anychoice>")

        # Player section
        tokens.append("<player>")

        # Active Pokemon
        if battle.active_pokemon:
            poke = battle.active_pokemon
            tokens.append(poke.species.lower())
            tokens.append(self._hp_bucket(poke.current_hp_fraction * 100))
            tokens.append(poke.status.name.lower() if poke.status else "nostatus")

            # Moves
            for move in poke.moves.values():
                tokens.append(move.id.lower())

        # Team (benched)
        for poke in battle.available_switches:
            tokens.append(poke.species.lower())
            tokens.append(self._hp_bucket(poke.current_hp_fraction * 100))
            tokens.append(poke.status.name.lower() if poke.status else "nostatus")

        # Opponent section
        tokens.append("<opponent>")

        if battle.opponent_active_pokemon:
            opp = battle.opponent_active_pokemon
            tokens.append(opp.species.lower())
            tokens.append(self._hp_bucket(opp.current_hp_fraction * 100))
            tokens.append(opp.status.name.lower() if opp.status else "nostatus")

        # Opponent team (what we've seen)
        for poke in battle.opponent_team.values():
            if poke != battle.opponent_active_pokemon:
                tokens.append(poke.species.lower())
                tokens.append(self._hp_bucket(poke.current_hp_fraction * 100))
                tokens.append(poke.status.name.lower() if poke.status else "nostatus")

        # Conditions
        tokens.append("<conditions>")

        # Weather
        if battle.weather:
            for weather in battle.weather:
                tokens.append(weather.name.lower())

        # Side conditions
        for condition in battle.side_conditions:
            tokens.append(condition.name.lower())
        for condition in battle.opponent_side_conditions:
            tokens.append(f"opp{condition.name.lower()}")

        if not battle.weather and not battle.side_conditions and not battle.opponent_side_conditions:
            tokens.append("noweather")

        # Tokenize
        text_obs = " ".join(tokens[:87])
        text_tokens = self.tokenizer.encode_observation(text_obs, max_length=87)

        return {
            "text_tokens": text_tokens.unsqueeze(0).unsqueeze(0).to(self.device),  # [1, 1, 87]
            "numerical_features": torch.zeros(1, 1, 48, device=self.device),  # [1, 1, 48]
            "prev_actions": torch.zeros(1, 1, dtype=torch.long, device=self.device),
            "prev_rewards": torch.zeros(1, 1, device=self.device),
            "turn_mask": torch.ones(1, 1, dtype=torch.bool, device=self.device),
        }

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
        return "hp100"

    def _get_action_mask(self, battle: Battle) -> torch.Tensor:
        """Create mask for legal actions"""
        mask = torch.zeros(self.num_actions, device=self.device)

        # Available moves (0-3)
        if battle.available_moves:
            for i, move in enumerate(battle.available_moves[:4]):
                mask[i] = 1.0

        # Available switches (4-8)
        if battle.available_switches:
            for i, poke in enumerate(battle.available_switches[:5]):
                mask[4 + i] = 1.0

        # Tera moves (9-12) - only if can terastallize
        if battle.can_tera and battle.available_moves:
            for i, move in enumerate(battle.available_moves[:4]):
                mask[9 + i] = 1.0

        # Ensure at least one action is valid
        if mask.sum() == 0:
            mask[0] = 1.0  # Default to first move

        return mask

    def _action_to_order(self, action_idx: int, battle: Battle):
        """Convert action index to poke-env order"""
        if action_idx < 4:
            # Move
            if action_idx < len(battle.available_moves):
                return self.create_order(battle.available_moves[action_idx])
        elif action_idx < 9:
            # Switch
            switch_idx = action_idx - 4
            if switch_idx < len(battle.available_switches):
                return self.create_order(battle.available_switches[switch_idx])
        elif action_idx < 13:
            # Tera + move
            move_idx = action_idx - 9
            if move_idx < len(battle.available_moves) and battle.can_tera:
                return self.create_order(battle.available_moves[move_idx], terastallize=True)

        # Fallback: random legal action
        return self.choose_random_move(battle)

    def choose_move(self, battle: Battle):
        """Main decision function called by poke-env"""
        # Convert battle to observation
        obs = self._battle_to_observation(battle)
        action_mask = self._get_action_mask(battle)

        # Run model inference
        with torch.no_grad():
            outputs = self.model(**obs)
            action_logits = outputs["action_logits"][0, 0, :self.num_actions]  # [num_actions]

        # Apply action mask (set illegal actions to -inf)
        action_logits = action_logits.masked_fill(action_mask == 0, float("-inf"))

        # Sample or argmax
        if self.deterministic:
            action_idx = action_logits.argmax().item()
        else:
            probs = F.softmax(action_logits / self.temperature, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()

        return self._action_to_order(action_idx, battle)


def create_agent(
    checkpoint_path: str,
    model_size: str = "base",
    battle_format: str = "gen9ou",
    **kwargs
) -> PokemonAIAgent:
    """Create a ready-to-battle agent from a checkpoint"""
    return PokemonAIAgent(
        checkpoint_path=checkpoint_path,
        model_size=model_size,
        battle_format=battle_format,
        **kwargs
    )
