"""
Pokemon Transformer - Main policy/value network architecture

This is a scaled-up causal transformer that:
1. Takes turn embeddings from TurnEncoder
2. Processes the full battle history with causal attention
3. Outputs action probabilities (actor) and value estimates (critic)

Designed for 1B+ parameters with modern training techniques:
- Flash Attention 2
- Rotary Position Embeddings
- SwiGLU activation
- Multi-query attention option
- Gradient checkpointing
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

from pokemon_ai.models.turn_encoder import TurnEncoder, TurnEncoderConfig, BatchedTurnEncoder


@dataclass
class PokemonTransformerConfig:
    """Configuration for the main Pokemon Transformer"""

    # Turn encoder config
    turn_encoder: TurnEncoderConfig = field(default_factory=TurnEncoderConfig)

    # Main transformer architecture
    hidden_dim: int = 2048  # Model dimension
    num_layers: int = 24  # Number of transformer layers
    num_heads: int = 16  # Attention heads
    num_kv_heads: Optional[int] = None  # For grouped-query attention (None = MHA)
    intermediate_dim: Optional[int] = None  # FFN dimension (default: 4 * hidden_dim)
    max_context_length: int = 256  # Maximum battle length in turns

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.0
    embed_dropout: float = 0.1

    # Output heads
    num_actions: int = 9  # 4 moves + 5 switches
    num_value_bins: int = 128  # For two-hot value encoding
    value_range: Tuple[float, float] = (-150.0, 150.0)  # Min/max return values
    num_gammas: int = 4  # Multiple discount factors
    gammas: Tuple[float, ...] = (0.9, 0.99, 0.999, 0.9999)

    # Training options
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    tie_word_embeddings: bool = False

    # Normalization
    rms_norm_eps: float = 1e-6
    use_pre_norm: bool = True

    def __post_init__(self):
        if self.intermediate_dim is None:
            # SwiGLU uses 2/3 factor
            self.intermediate_dim = int(self.hidden_dim * 4 * 2 / 3)
            # Round to nearest multiple of 256 for efficiency
            self.intermediate_dim = ((self.intermediate_dim + 255) // 256) * 256

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings with dynamic cache"""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PokemonAttention(nn.Module):
    """
    Multi-head attention with:
    - Grouped-query attention support
    - Rotary embeddings
    - Flash attention when available
    - KV caching for inference
    """

    def __init__(self, config: PokemonTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_context_length)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Get rotary embeddings
        cos, sin = self.rotary(seq_len)

        # Handle KV cache for inference
        if past_key_value is not None:
            past_k, past_v = past_key_value
            # Apply rotary to new positions only
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if use_cache:
            past_key_value = (k, v)
        else:
            past_key_value = None

        # Repeat KV heads for grouped-query attention
        if self.num_key_value_groups > 1:
            k = repeat(k, "b h s d -> b (h g) s d", g=self.num_key_value_groups)
            v = repeat(v, "b h s d -> b (h g) s d", g=self.num_key_value_groups)

        # Compute attention
        if self.config.use_flash_attention and x.is_cuda and not use_cache:
            try:
                from flash_attn import flash_attn_func
                # Flash attention expects (batch, seq, heads, dim)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.config.attention_dropout if self.training else 0.0,
                    causal=True,
                )
                attn_output = attn_output.reshape(batch_size, seq_len, -1)
            except ImportError:
                attn_output = self._standard_attention(q, k, v, attention_mask)
        else:
            attn_output = self._standard_attention(q, k, v, attention_mask)

        output = self.o_proj(attn_output)
        return output, past_key_value

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_heads, q_len, head_dim = q.shape
        kv_len = k.shape[2]

        scale = head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), device=q.device),
                diagonal=kv_len - q_len + 1,
            )
            attn_weights = attn_weights + causal_mask
        else:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q_len, -1)

        return attn_output


class PokemonMLP(nn.Module):
    """SwiGLU MLP block"""

    def __init__(self, config: PokemonTransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class PokemonTransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture"""

    def __init__(self, config: PokemonTransformerConfig, layer_idx: int):
        super().__init__()
        self.attention = PokemonAttention(config, layer_idx)
        self.mlp = PokemonMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention with residual
        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.attention(x, attention_mask, position_ids, past_key_value, use_cache)
        x = self.dropout(x)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = residual + x

        return x, present_key_value


class TwoHotEncoder(nn.Module):
    """Two-hot encoding for value prediction (improves stability)"""

    def __init__(self, num_bins: int, value_range: Tuple[float, float]):
        super().__init__()
        self.num_bins = num_bins
        self.min_value, self.max_value = value_range

        # Create bin edges
        self.register_buffer(
            "bin_edges",
            torch.linspace(self.min_value, self.max_value, num_bins)
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Convert continuous values to two-hot distribution"""
        values = torch.clamp(values, self.min_value, self.max_value)

        # Find bin indices
        bin_width = (self.max_value - self.min_value) / (self.num_bins - 1)
        normalized = (values - self.min_value) / bin_width
        lower_idx = torch.floor(normalized).long()
        upper_idx = torch.ceil(normalized).long()

        # Clamp indices
        lower_idx = torch.clamp(lower_idx, 0, self.num_bins - 1)
        upper_idx = torch.clamp(upper_idx, 0, self.num_bins - 1)

        # Calculate weights
        upper_weight = normalized - lower_idx.float()
        lower_weight = 1.0 - upper_weight

        # Create two-hot encoding
        two_hot = torch.zeros(*values.shape, self.num_bins, device=values.device)
        two_hot.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        two_hot.scatter_add_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))

        return two_hot

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits back to continuous values"""
        probs = F.softmax(logits, dim=-1)
        values = torch.sum(probs * self.bin_edges, dim=-1)
        return values


class ActorHead(nn.Module):
    """Actor head for action prediction"""

    def __init__(self, config: PokemonTransformerConfig):
        super().__init__()
        self.config = config

        self.layers = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 4, config.num_actions),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # hidden_states: [batch, seq_len, hidden_dim] or [batch, hidden_dim]
        logits = self.layers(hidden_states)
        # logits: [batch, seq_len, num_actions] or [batch, num_actions]

        # Mask invalid actions
        if action_mask is not None:
            # action_mask is [batch, num_actions] - need to broadcast properly
            if logits.dim() == 3 and action_mask.dim() == 2:
                # Expand mask to [batch, 1, num_actions] for broadcasting
                action_mask = action_mask.unsqueeze(1)
            logits = logits.masked_fill(~action_mask, float("-inf"))

        return logits


class CriticHead(nn.Module):
    """Critic head with two-hot value prediction and multiple discount factors"""

    def __init__(self, config: PokemonTransformerConfig):
        super().__init__()
        self.config = config
        self.num_gammas = config.num_gammas

        self.shared = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Separate output for each discount factor
        self.value_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim // 4, config.num_value_bins)
            for _ in range(config.num_gammas)
        ])

        self.two_hot = TwoHotEncoder(config.num_value_bins, config.value_range)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_out = self.shared(hidden_states)

        outputs = {}
        for i, head in enumerate(self.value_heads):
            gamma = self.config.gammas[i]
            logits = head(shared_out)
            value = self.two_hot.decode(logits)
            outputs[f"value_gamma_{gamma}"] = value
            outputs[f"value_logits_gamma_{gamma}"] = logits

        return outputs


class PokemonTransformer(nn.Module):
    """
    Main Pokemon Transformer model.

    Architecture:
    1. TurnEncoder processes each turn's observation
    2. Input projection to hidden_dim
    3. Causal Transformer processes battle history
    4. Actor and Critic heads for RL

    Supports:
    - Gradient checkpointing for memory efficiency
    - KV caching for inference
    - Flash Attention 2
    - Multi-gamma value estimation
    """

    def __init__(self, config: PokemonTransformerConfig):
        super().__init__()
        self.config = config

        # Turn encoder
        self.turn_encoder = BatchedTurnEncoder(config.turn_encoder)

        # Input projection from turn encoder to transformer
        self.input_projection = nn.Linear(
            config.turn_encoder.hidden_dim,
            config.hidden_dim,
            bias=False,
        )

        # Dropout after projection
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            PokemonTransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

        # Output heads
        self.actor = ActorHead(config)
        self.critic = CriticHead(config)

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special initialization to output projections
        for layer in self.layers:
            layer.attention.o_proj.weight.data.mul_(1 / math.sqrt(2 * config.num_layers))
            layer.mlp.down_proj.weight.data.mul_(1 / math.sqrt(2 * config.num_layers))

        # Enable gradient checkpointing if configured
        self.gradient_checkpointing = config.use_gradient_checkpointing

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.turn_encoder.encoder.text_embedding.weight.numel()
        return n_params

    def forward(
        self,
        text_tokens: torch.Tensor,  # [batch, max_turns, num_text_tokens]
        numerical_features: torch.Tensor,  # [batch, max_turns, num_numerical_tokens]
        prev_actions: torch.Tensor,  # [batch, max_turns]
        prev_rewards: torch.Tensor,  # [batch, max_turns]
        turn_mask: Optional[torch.Tensor] = None,  # [batch, max_turns]
        action_mask: Optional[torch.Tensor] = None,  # [batch, max_turns, num_actions]
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size, max_turns = text_tokens.shape[:2]
        device = text_tokens.device

        # Encode turns
        turn_embeds = self.turn_encoder(
            text_tokens, numerical_features, prev_actions, prev_rewards, turn_mask
        )  # [batch, max_turns, turn_hidden_dim]

        # Project to transformer hidden dim
        hidden_states = self.input_projection(turn_embeds)
        hidden_states = self.embed_dropout(hidden_states)

        # Create position IDs if not provided
        if position_ids is None:
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]
                position_ids = torch.arange(
                    past_length, past_length + max_turns, device=device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(max_turns, device=device).unsqueeze(0).expand(batch_size, -1)

        # Create causal attention mask
        # Shape needs to be [batch, 1, seq_len, seq_len] for broadcasting with attn weights
        attention_mask = None
        if turn_mask is not None:
            # Causal mask: [seq_len, seq_len]
            causal_mask = torch.triu(
                torch.full((max_turns, max_turns), float("-inf"), device=device),
                diagonal=1,
            )
            # Padding mask: positions where turn_mask is False should be masked
            # [batch, seq_len] -> [batch, 1, 1, seq_len]
            # Note: We use masked_fill instead of multiplication to avoid 0 * -inf = NaN
            padding_mask = torch.zeros(batch_size, max_turns, device=device, dtype=hidden_states.dtype)
            padding_mask = padding_mask.masked_fill(~turn_mask, float("-inf"))
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            # Combine: [1, 1, seq_len, seq_len] + [batch, 1, 1, seq_len]
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0) + padding_mask

        # Process through transformer layers
        presents = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, present = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, present = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )

            if presents is not None:
                presents.append(present)

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # Get action mask for the last position if not provided
        if action_mask is not None:
            last_action_mask = action_mask[:, -1, :]
        else:
            last_action_mask = None

        # Actor and critic outputs (typically for last position)
        actor_logits = self.actor(hidden_states, last_action_mask)
        critic_outputs = self.critic(hidden_states)

        outputs = {
            "actor_logits": actor_logits,
            "hidden_states": hidden_states,
            **critic_outputs,
        }

        if use_cache:
            outputs["past_key_values"] = presents

        return outputs

    @torch.no_grad()
    def get_action(
        self,
        text_tokens: torch.Tensor,
        numerical_features: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
        gamma: float = 0.999,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get action from the policy (inference mode)"""
        outputs = self.forward(
            text_tokens, numerical_features, prev_actions, prev_rewards,
            action_mask=action_mask,
            use_cache=False,
        )

        logits = outputs["actor_logits"][:, -1, :]  # Last position

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Get value for selected gamma
        value_key = f"value_gamma_{gamma}"
        value = outputs[value_key][:, -1] if value_key in outputs else None

        info = {
            "action_probs": F.softmax(logits, dim=-1),
            "value": value,
        }

        return action, info


def create_pokemon_transformer(
    size: str = "base",
    **kwargs,
) -> PokemonTransformer:
    """
    Factory function to create Pokemon Transformer with predefined sizes.

    Sizes:
    - "small": ~50M params (for debugging)
    - "base": ~200M params (matches paper)
    - "large": ~500M params
    - "xl": ~1B params
    - "xxl": ~2B params
    """
    configs = {
        "small": {
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "turn_encoder": TurnEncoderConfig(token_dim=128, hidden_dim=256, num_layers=2),
        },
        "base": {
            "hidden_dim": 1024,
            "num_layers": 16,
            "num_heads": 16,
            "turn_encoder": TurnEncoderConfig(token_dim=192, hidden_dim=384, num_layers=3),
        },
        "large": {
            "hidden_dim": 1536,
            "num_layers": 24,
            "num_heads": 16,
            "turn_encoder": TurnEncoderConfig(token_dim=256, hidden_dim=512, num_layers=4),
        },
        "xl": {
            "hidden_dim": 2048,
            "num_layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,  # GQA for efficiency
            "turn_encoder": TurnEncoderConfig(token_dim=320, hidden_dim=640, num_layers=5),
        },
        "xxl": {
            "hidden_dim": 2560,
            "num_layers": 40,
            "num_heads": 40,
            "num_kv_heads": 8,
            "turn_encoder": TurnEncoderConfig(token_dim=384, hidden_dim=768, num_layers=6),
        },
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    config_dict = configs[size]
    config_dict.update(kwargs)

    # Create turn encoder config
    turn_encoder_config = config_dict.pop("turn_encoder")
    config = PokemonTransformerConfig(turn_encoder=turn_encoder_config, **config_dict)

    return PokemonTransformer(config)
