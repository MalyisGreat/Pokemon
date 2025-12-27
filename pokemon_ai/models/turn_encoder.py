"""
Turn Encoder - Processes individual turn observations into embeddings

This module handles the multi-modal input at each timestep:
- Text tokens (Pokemon names, moves, items, abilities, etc.)
- Numerical features (HP, stats, boosts, etc.)
- Previous action and reward
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class TurnEncoderConfig:
    """Configuration for Turn Encoder"""

    # Vocabulary
    vocab_size: int = 8192  # Pokemon vocabulary size
    num_text_tokens: int = 87  # Fixed text sequence length
    num_numerical_tokens: int = 48  # Numerical features
    num_summary_tokens: int = 8  # Learnable summary tokens

    # Architecture
    token_dim: int = 256  # Embedding dimension per token
    hidden_dim: int = 512  # Hidden dimension after encoding
    num_layers: int = 4  # Number of transformer layers
    num_heads: int = 8  # Attention heads
    dropout: float = 0.1

    # Action/reward embedding
    num_actions: int = 9  # 4 moves + 5 switches
    action_embed_dim: int = 64

    # Extras
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TurnEncoderAttention(nn.Module):
    """Multi-head attention for turn encoder with optional flash attention"""

    def __init__(self, config: TurnEncoderConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.token_dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.token_dim, config.token_dim)
        self.k_proj = nn.Linear(config.token_dim, config.token_dim)
        self.v_proj = nn.Linear(config.token_dim, config.token_dim)
        self.o_proj = nn.Linear(config.token_dim, config.token_dim)

        self.dropout = nn.Dropout(config.dropout)

        if config.use_rotary_embeddings:
            self.rotary = RotaryEmbedding(self.head_dim)
        else:
            self.rotary = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply rotary embeddings
        if self.rotary is not None:
            cos, sin = self.rotary(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention
        if self.config.use_flash_attention and x.is_cuda:
            try:
                from flash_attn import flash_attn_func
                # Flash attention expects (batch, seq, heads, dim)
                q = rearrange(q, "b h s d -> b s h d")
                k = rearrange(k, "b h s d -> b s h d")
                v = rearrange(v, "b h s d -> b s h d")
                attn_output = flash_attn_func(q, k, v, dropout_p=self.config.dropout if self.training else 0.0)
                attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
            except ImportError:
                attn_output = self._standard_attention(q, k, v, attention_mask)
        else:
            attn_output = self._standard_attention(q, k, v, attention_mask)

        return self.o_proj(attn_output)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")

        return attn_output


class TurnEncoderMLP(nn.Module):
    """MLP block with SwiGLU activation"""

    def __init__(self, config: TurnEncoderConfig):
        super().__init__()
        hidden_dim = int(config.token_dim * 4 * 2 / 3)  # SwiGLU uses 2/3 factor

        self.gate_proj = nn.Linear(config.token_dim, hidden_dim)
        self.up_proj = nn.Linear(config.token_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, config.token_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class TurnEncoderLayer(nn.Module):
    """Single transformer layer for turn encoder"""

    def __init__(self, config: TurnEncoderConfig):
        super().__init__()
        self.attention = TurnEncoderAttention(config)
        self.mlp = TurnEncoderMLP(config)
        self.ln1 = nn.LayerNorm(config.token_dim)
        self.ln2 = nn.LayerNorm(config.token_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm architecture
        residual = x
        x = self.ln1(x)
        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = residual + x

        return x


class TurnEncoder(nn.Module):
    """
    Encodes a single turn observation into a fixed-size representation.

    Input format per turn:
    - text_tokens: [batch, num_text_tokens] - tokenized Pokemon vocabulary
    - numerical_features: [batch, num_numerical_tokens] - HP, stats, etc.
    - prev_action: [batch] - previous action index
    - prev_reward: [batch] - previous reward value

    Output: [batch, hidden_dim] - turn representation
    """

    def __init__(self, config: TurnEncoderConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.text_embedding = nn.Embedding(config.vocab_size, config.token_dim)
        self.numerical_projection = nn.Linear(1, config.token_dim)

        # Summary tokens (learnable)
        self.summary_tokens = nn.Parameter(
            torch.randn(1, config.num_summary_tokens, config.token_dim) * 0.02
        )

        # Action and reward embeddings
        self.action_embedding = nn.Embedding(config.num_actions + 1, config.action_embed_dim)  # +1 for no-op
        self.reward_projection = nn.Linear(1, config.action_embed_dim)
        self.action_reward_projection = nn.Linear(config.action_embed_dim * 2, config.token_dim)

        # Position embeddings for different token types
        total_tokens = config.num_text_tokens + config.num_numerical_tokens + config.num_summary_tokens + 1
        self.token_type_embedding = nn.Embedding(4, config.token_dim)  # text, numerical, summary, action/reward

        # Transformer layers
        self.layers = nn.ModuleList([
            TurnEncoderLayer(config) for _ in range(config.num_layers)
        ])

        # Final projection
        self.final_ln = nn.LayerNorm(config.token_dim)
        self.output_projection = nn.Linear(
            config.token_dim * config.num_summary_tokens,
            config.hidden_dim
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        text_tokens: torch.Tensor,  # [batch, num_text_tokens]
        numerical_features: torch.Tensor,  # [batch, num_numerical_tokens]
        prev_action: torch.Tensor,  # [batch]
        prev_reward: torch.Tensor,  # [batch]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = text_tokens.shape[0]
        device = text_tokens.device

        # Clamp indices to valid range to prevent index out of bounds errors
        text_tokens = text_tokens.clamp(0, self.config.vocab_size - 1)
        prev_action = prev_action.clamp(0, self.config.num_actions)

        # Embed text tokens
        text_embeds = self.text_embedding(text_tokens)  # [batch, num_text, token_dim]

        # Embed numerical features
        numerical_embeds = self.numerical_projection(
            numerical_features.unsqueeze(-1)
        )  # [batch, num_numerical, token_dim]

        # Expand summary tokens
        summary_embeds = self.summary_tokens.expand(batch_size, -1, -1)  # [batch, num_summary, token_dim]

        # Embed action and reward
        action_embed = self.action_embedding(prev_action)  # [batch, action_embed_dim]
        reward_embed = self.reward_projection(prev_reward.unsqueeze(-1))  # [batch, action_embed_dim]
        action_reward_embed = self.action_reward_projection(
            torch.cat([action_embed, reward_embed], dim=-1)
        ).unsqueeze(1)  # [batch, 1, token_dim]

        # Concatenate all embeddings
        x = torch.cat([
            text_embeds,
            numerical_embeds,
            summary_embeds,
            action_reward_embed,
        ], dim=1)  # [batch, total_tokens, token_dim]

        # Add token type embeddings
        token_types = torch.cat([
            torch.zeros(batch_size, self.config.num_text_tokens, device=device, dtype=torch.long),
            torch.ones(batch_size, self.config.num_numerical_tokens, device=device, dtype=torch.long),
            torch.full((batch_size, self.config.num_summary_tokens), 2, device=device, dtype=torch.long),
            torch.full((batch_size, 1), 3, device=device, dtype=torch.long),
        ], dim=1)
        x = x + self.token_type_embedding(token_types)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.final_ln(x)

        # Extract summary tokens and project to output
        summary_start = self.config.num_text_tokens + self.config.num_numerical_tokens
        summary_end = summary_start + self.config.num_summary_tokens
        summary_output = x[:, summary_start:summary_end, :]  # [batch, num_summary, token_dim]

        # Flatten and project
        summary_flat = rearrange(summary_output, "b s d -> b (s d)")
        output = self.output_projection(summary_flat)  # [batch, hidden_dim]

        return output


class BatchedTurnEncoder(nn.Module):
    """
    Wrapper that processes multiple turns in parallel for efficiency.

    Input: [batch, max_turns, ...]
    Output: [batch, max_turns, hidden_dim]
    """

    def __init__(self, config: TurnEncoderConfig):
        super().__init__()
        self.encoder = TurnEncoder(config)
        self.config = config

    def forward(
        self,
        text_tokens: torch.Tensor,  # [batch, max_turns, num_text_tokens]
        numerical_features: torch.Tensor,  # [batch, max_turns, num_numerical_tokens]
        prev_actions: torch.Tensor,  # [batch, max_turns]
        prev_rewards: torch.Tensor,  # [batch, max_turns]
        turn_mask: Optional[torch.Tensor] = None,  # [batch, max_turns] - which turns are valid
    ) -> torch.Tensor:
        batch_size, max_turns = text_tokens.shape[:2]

        # Reshape to process all turns at once
        text_flat = rearrange(text_tokens, "b t s -> (b t) s")
        numerical_flat = rearrange(numerical_features, "b t s -> (b t) s")
        actions_flat = rearrange(prev_actions, "b t -> (b t)")
        rewards_flat = rearrange(prev_rewards, "b t -> (b t)")

        # Encode all turns
        encoded = self.encoder(text_flat, numerical_flat, actions_flat, rewards_flat)

        # Reshape back
        encoded = rearrange(encoded, "(b t) d -> b t d", b=batch_size, t=max_turns)

        # Mask invalid turns
        if turn_mask is not None:
            encoded = encoded * turn_mask.unsqueeze(-1)

        return encoded
