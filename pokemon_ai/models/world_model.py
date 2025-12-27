"""
World Model - Learns Pokemon battle dynamics for planning

The world model predicts:
1. Next state given (current state, player action, opponent action)
2. Opponent's action distribution given current state
3. Reward for the transition

This enables MCTS/planning at inference time.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pokemon_ai.models.turn_encoder import TurnEncoderConfig


@dataclass
class WorldModelConfig:
    """Configuration for World Model"""

    # Input dimensions (from turn encoder)
    state_dim: int = 512  # Hidden dim from turn encoder
    action_dim: int = 9  # Number of possible actions

    # Architecture
    hidden_dim: int = 1024
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    # Prediction heads
    num_reward_bins: int = 64
    reward_range: Tuple[float, float] = (-10.0, 110.0)

    # State prediction
    predict_delta: bool = True  # Predict delta vs full state
    state_loss_type: str = "mse"  # "mse" or "cosine"

    # Opponent modeling
    model_opponent: bool = True


class TransformerBlock(nn.Module):
    """Standard transformer block for world model"""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with pre-norm
        residual = x
        x = self.ln1(x)
        x, _ = self.attention(x, x, x, attn_mask=mask)
        x = residual + x

        # MLP with pre-norm
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class StateEncoder(nn.Module):
    """Encodes the current battle state for world model input"""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # Project state to hidden dim
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)

        # Action embeddings
        self.player_action_embed = nn.Embedding(config.action_dim + 1, config.hidden_dim // 2)  # +1 for unknown
        self.opponent_action_embed = nn.Embedding(config.action_dim + 1, config.hidden_dim // 2)

        # Combine state and actions
        self.combine = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(
        self,
        state: torch.Tensor,  # [batch, state_dim]
        player_action: torch.Tensor,  # [batch]
        opponent_action: Optional[torch.Tensor] = None,  # [batch] or None
    ) -> torch.Tensor:
        state_embed = self.state_proj(state)

        player_embed = self.player_action_embed(player_action)

        if opponent_action is not None:
            opponent_embed = self.opponent_action_embed(opponent_action)
        else:
            # Use learned "unknown" embedding
            unknown_idx = torch.full_like(player_action, self.config.action_dim)
            opponent_embed = self.opponent_action_embed(unknown_idx)

        # Concatenate and combine
        action_embed = torch.cat([player_embed, opponent_embed], dim=-1)
        combined = torch.cat([state_embed, action_embed], dim=-1)

        return self.combine(combined)


class RewardPredictor(nn.Module):
    """Predicts reward as a categorical distribution (two-hot style)"""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.num_bins = config.num_reward_bins
        self.min_val, self.max_val = config.reward_range

        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.num_reward_bins),
        )

        # Bin centers for decoding
        self.register_buffer(
            "bin_centers",
            torch.linspace(self.min_val, self.max_val, config.num_reward_bins)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reward logits"""
        return self.net(x)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to scalar reward"""
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.bin_centers, dim=-1)

    def encode_target(self, rewards: torch.Tensor) -> torch.Tensor:
        """Create two-hot target from scalar rewards"""
        rewards = torch.clamp(rewards, self.min_val, self.max_val)

        bin_width = (self.max_val - self.min_val) / (self.num_bins - 1)
        normalized = (rewards - self.min_val) / bin_width

        lower_idx = torch.floor(normalized).long()
        upper_idx = torch.ceil(normalized).long()
        lower_idx = torch.clamp(lower_idx, 0, self.num_bins - 1)
        upper_idx = torch.clamp(upper_idx, 0, self.num_bins - 1)

        upper_weight = normalized - lower_idx.float()
        lower_weight = 1.0 - upper_weight

        target = torch.zeros(*rewards.shape, self.num_bins, device=rewards.device)
        target.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        target.scatter_add_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))

        return target


class OpponentActionPredictor(nn.Module):
    """Predicts opponent's action distribution"""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.action_dim),
        )

    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.net(x)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return logits


class StatePredictor(nn.Module):
    """Predicts next state (or state delta)"""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TerminationPredictor(nn.Module):
    """Predicts if the battle ends"""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WorldModel(nn.Module):
    """
    Complete World Model for Pokemon battles.

    Given: current state embedding + player action + (optional) opponent action
    Predicts:
    - Next state embedding
    - Reward
    - Done flag
    - Opponent action distribution (if not provided)
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # State encoder
        self.state_encoder = StateEncoder(config)

        # Transformer backbone
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_dim)

        # Prediction heads
        self.state_predictor = StatePredictor(config)
        self.reward_predictor = RewardPredictor(config)
        self.termination_predictor = TerminationPredictor(config)

        if config.model_opponent:
            self.opponent_predictor = OpponentActionPredictor(config)
        else:
            self.opponent_predictor = None

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        state: torch.Tensor,  # [batch, state_dim]
        player_action: torch.Tensor,  # [batch]
        opponent_action: Optional[torch.Tensor] = None,  # [batch]
        opponent_action_mask: Optional[torch.Tensor] = None,  # [batch, num_actions]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through world model.

        If opponent_action is None, first predicts opponent action, then uses it.
        """
        outputs = {}

        # Predict opponent action if not provided
        if opponent_action is None and self.opponent_predictor is not None:
            # Encode state without opponent action for opponent prediction
            state_only = self.state_encoder(state, player_action, opponent_action=None)
            opp_logits = self.opponent_predictor(state_only, opponent_action_mask)
            outputs["opponent_action_logits"] = opp_logits

            # Sample opponent action for dynamics prediction
            opp_probs = F.softmax(opp_logits, dim=-1)
            opponent_action = torch.multinomial(opp_probs, num_samples=1).squeeze(-1)
            outputs["predicted_opponent_action"] = opponent_action

        # Encode full state + actions
        encoded = self.state_encoder(state, player_action, opponent_action)
        encoded = encoded.unsqueeze(1)  # Add sequence dim for transformer

        # Process through transformer
        for layer in self.layers:
            encoded = layer(encoded)
        encoded = self.final_norm(encoded)
        encoded = encoded.squeeze(1)  # Remove sequence dim

        # Predictions
        state_delta = self.state_predictor(encoded)
        if self.config.predict_delta:
            next_state = state + state_delta
        else:
            next_state = state_delta

        reward_logits = self.reward_predictor(encoded)
        reward = self.reward_predictor.decode(reward_logits)

        done_logits = self.termination_predictor(encoded)
        done_prob = torch.sigmoid(done_logits)

        outputs.update({
            "next_state": next_state,
            "state_delta": state_delta,
            "reward_logits": reward_logits,
            "reward": reward,
            "done_logits": done_logits,
            "done_prob": done_prob,
        })

        return outputs

    def compute_loss(
        self,
        state: torch.Tensor,
        player_action: torch.Tensor,
        opponent_action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        opponent_action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses"""
        outputs = self.forward(state, player_action, opponent_action=None, opponent_action_mask=opponent_action_mask)

        losses = {}

        # State prediction loss
        if self.config.state_loss_type == "mse":
            state_loss = F.mse_loss(outputs["next_state"], next_state)
        else:  # cosine
            state_loss = 1 - F.cosine_similarity(outputs["next_state"], next_state, dim=-1).mean()
        losses["state_loss"] = state_loss

        # Reward prediction loss (cross-entropy on two-hot)
        reward_target = self.reward_predictor.encode_target(reward)
        reward_loss = F.cross_entropy(outputs["reward_logits"], reward_target)
        losses["reward_loss"] = reward_loss

        # Termination loss
        done_loss = F.binary_cross_entropy_with_logits(outputs["done_logits"], done.float())
        losses["done_loss"] = done_loss

        # Opponent prediction loss
        if self.opponent_predictor is not None:
            opp_loss = F.cross_entropy(outputs["opponent_action_logits"], opponent_action)
            losses["opponent_loss"] = opp_loss

        # Total loss
        total_loss = (
            state_loss +
            0.5 * reward_loss +
            0.1 * done_loss +
            (0.5 * losses.get("opponent_loss", 0.0))
        )
        losses["total_loss"] = total_loss

        return losses

    @torch.no_grad()
    def rollout(
        self,
        initial_state: torch.Tensor,
        player_actions: torch.Tensor,  # [batch, horizon]
        opponent_action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform imagined rollout for planning.

        Returns predicted states, rewards, and done flags for the trajectory.
        """
        batch_size, horizon = player_actions.shape
        device = initial_state.device

        states = [initial_state]
        rewards = []
        dones = []
        opponent_actions = []

        current_state = initial_state

        for t in range(horizon):
            outputs = self.forward(
                current_state,
                player_actions[:, t],
                opponent_action=None,
                opponent_action_mask=opponent_action_mask,
            )

            states.append(outputs["next_state"])
            rewards.append(outputs["reward"])
            dones.append(outputs["done_prob"])

            if "predicted_opponent_action" in outputs:
                opponent_actions.append(outputs["predicted_opponent_action"])

            current_state = outputs["next_state"]

        return {
            "states": torch.stack(states, dim=1),  # [batch, horizon+1, state_dim]
            "rewards": torch.stack(rewards, dim=1),  # [batch, horizon]
            "dones": torch.stack(dones, dim=1),  # [batch, horizon]
            "opponent_actions": torch.stack(opponent_actions, dim=1) if opponent_actions else None,
        }
