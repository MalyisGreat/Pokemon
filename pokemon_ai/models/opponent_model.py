"""
Opponent Model - Explicit opponent team and policy inference

Tracks beliefs about:
1. Opponent's team composition (which Pokemon, moves, items, abilities)
2. Opponent's policy tendencies
"""

from dataclasses import dataclass
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OpponentModelConfig:
    """Configuration for Opponent Model"""
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    num_pokemon: int = 1000  # Vocabulary of Pokemon species
    num_moves: int = 800  # Vocabulary of moves
    num_items: int = 400  # Vocabulary of items
    num_abilities: int = 300  # Vocabulary of abilities
    num_actions: int = 9
    max_team_size: int = 6
    dropout: float = 0.1


class OpponentModel(nn.Module):
    """
    Models opponent's hidden information and tendencies.

    Given the battle history, predicts:
    - Unrevealed Pokemon on opponent's team
    - Unrevealed moves/items/abilities
    - Opponent's action distribution
    """

    def __init__(self, config: OpponentModelConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Transformer for processing history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Team prediction heads
        self.pokemon_predictor = nn.Linear(config.hidden_dim, config.num_pokemon)
        self.move_predictor = nn.Linear(config.hidden_dim, config.num_moves)
        self.item_predictor = nn.Linear(config.hidden_dim, config.num_items)
        self.ability_predictor = nn.Linear(config.hidden_dim, config.num_abilities)

        # Action prediction
        self.action_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.num_actions),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.input_proj(hidden_states)
        x = self.transformer(x, src_key_padding_mask=mask)

        # Use last position for predictions
        last = x[:, -1, :]

        return {
            "pokemon_logits": self.pokemon_predictor(last),
            "move_logits": self.move_predictor(last),
            "item_logits": self.item_predictor(last),
            "ability_logits": self.ability_predictor(last),
            "action_logits": self.action_predictor(last),
        }
