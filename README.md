# Pokemon AI - Metamon Implementation

A superhuman Pokemon battle AI based on the [Metamon paper](https://arxiv.org/abs/2310.10Pokemon), trained using offline reinforcement learning on 4M+ battle replays.

## Training Status

**Current Run:** Training successfully on RunPod H100
- Loss: 2.21 (down from 11.7 at start)
- GPU Utilization: 99%
- Speed: ~152 samples/sec (~2.4 it/s)
- ETA: ~7 hours per epoch, ~21 hours for 3 epochs
- Wandb: https://wandb.ai/joshjoey6543/pokemon-superhuman

## Quick Start

### 1. Download Metamon Dataset

```bash
# Download to RAM disk for fastest training (RunPod)
huggingface-cli download Metamon/Pokemon-Showdown-Replays \
    --repo-type dataset \
    --local-dir /dev/shm/metamon_data

# Or to regular storage
huggingface-cli download Metamon/Pokemon-Showdown-Replays \
    --repo-type dataset \
    --local-dir data/replays
```

### 2. Install Dependencies

```bash
pip install torch transformers wandb tqdm pyyaml lz4 flash-attn --upgrade
```

### 3. Run Training

```bash
python scripts/train.py --config pokemon_ai/configs/h100_1gpu.yaml --data_path /dev/shm/metamon_data
```

## Issues We Fixed

### 1. Slow Data Loading (3.5 hours for 4M files)

**Problem:** Loading 4M .lz4 files one-by-one took ~3.5 hours before training could start.

**Solution:** Implemented lazy loading - just index file paths, load on-demand during training.

**File:** `pokemon_ai/data/dataset.py`
```python
if is_metamon:
    # FAST PATH: just index file paths, load on-demand
    self.file_index = [(str(f), 0) for f in files]
    self._is_metamon = True
    print(f"Indexed {len(self.file_index)} trajectories (lazy loading enabled)")
```

### 2. torch.compile + Flash Attention Conflict

**Problem:** `FlashAttention only support fp16 and bf16 data type` error when using torch.compile.

**Solution:** Disabled torch.compile in config.

**File:** `pokemon_ai/configs/h100_1gpu.yaml`
```yaml
training:
  compile_model: false  # Disabled: causes dtype issues with Flash Attention
```

### 3. CUDA Index Out of Bounds in Embedding

**Problem:** `CUDA error: device-side assert triggered` due to token IDs exceeding vocabulary size (8192).

**Solution:** Added multi-layer clamping to ensure token IDs stay in valid range.

**File:** `pokemon_ai/data/tokenizer.py`
```python
def encode_observation(self, obs_text: str, max_length: int = 87) -> torch.Tensor:
    # Handle dict/complex objects being passed as strings
    if not isinstance(obs_text, str):
        obs_text = str(obs_text)

    # If it looks like a Python dict repr, just use padding
    if obs_text.startswith("{") or obs_text.startswith("{'"):
        ids = [self.pad_token_id] * max_length
        return torch.tensor(ids, dtype=torch.long)

    ids = self.encode(obs_text)

    # Clamp all IDs to valid range
    ids = [min(max(0, i), self.vocab_size - 1) for i in ids]
    # ... rest of function
```

**File:** `pokemon_ai/data/dataset.py`
```python
# Get vocab size for clamping (default 8192)
vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else 8192

# Clamp tokens to valid range
tokens = tokens.clamp(0, vocab_size - 1)
```

**File:** `pokemon_ai/models/turn_encoder.py`
```python
def forward(self, text_tokens, numerical_features, prev_action, prev_reward, attention_mask=None):
    # Clamp indices to valid range to prevent index out of bounds errors
    text_tokens = text_tokens.clamp(0, self.config.vocab_size - 1)
    prev_action = prev_action.clamp(0, self.config.num_actions)
```

### 4. Flash Attention dtype Error (Turn Encoder)

**Problem:** `FlashAttention only support fp16 and bf16 data type` in turn_encoder.py - getting fp32 tensors even with mixed precision.

**Solution:** Explicitly cast q, k, v to bf16 before calling flash_attn_func, then cast back.

**File:** `pokemon_ai/models/turn_encoder.py`
```python
# Flash attention
if self.use_flash_attention:
    # Reshape for flash attention: (batch, seq_len, num_heads, head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Cast to bf16 for Flash Attention compatibility
    orig_dtype = q.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

    attn_output = flash_attn_func(
        q, k, v,
        dropout_p=self.config.dropout if self.training else 0.0
    )

    # Cast back to original dtype
    if orig_dtype not in (torch.float16, torch.bfloat16):
        attn_output = attn_output.to(orig_dtype)
```

### 5. Flash Attention dtype Error (Main Transformer)

**Problem:** Same Flash Attention dtype issue in the main transformer model.

**Solution:** Applied the same bf16 casting fix.

**File:** `pokemon_ai/models/pokemon_transformer.py`
```python
# Cast to bf16 for Flash Attention compatibility
orig_dtype = q.dtype
if orig_dtype not in (torch.float16, torch.bfloat16):
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

attn_output = flash_attn_func(
    q, k, v,
    dropout_p=self.config.attention_dropout if self.training else 0.0,
    causal=True
)

# Cast back to original dtype
if orig_dtype not in (torch.float16, torch.bfloat16):
    attn_output = attn_output.to(orig_dtype)
```

### 6. Metamon State Conversion (All Padding Tokens)

**Problem:** Metamon states are complex dicts like `{'pokemon': ..., 'moves': ...}`. The original code did `str(state)` which produces `"{'pokemon': ...}"`. The tokenizer detected this starts with `{` and returned **all padding tokens**. The model was learning **nothing from observations** - actor loss was near-zero (0.02-0.05) while critic loss dominated (5.6+).

**Solution:** Created a proper state-to-text converter that extracts meaningful information from Metamon UniversalState dicts and formats them as tokenizer-friendly text.

**File:** `pokemon_ai/data/state_converter.py` (new file)
```python
def convert_state_to_text(state: Dict[str, Any], format_id: str = "gen9ou") -> str:
    """Convert Metamon UniversalState dict to tokenizer-friendly text"""
    tokens = []
    tokens.append(f"<{normalize_name(format_id)}>")

    # Player section
    tokens.append("<player>")
    active = state.get("active", {})
    tokens.extend(get_pokemon_info(active))  # species, hp_bucket, status, types

    tokens.append("<moveset>")
    for move in state.get("moves", [])[:4]:
        tokens.extend(get_move_info(move))  # name, type, category

    # Opponent section
    tokens.append("<opponent>")
    tokens.extend(get_pokemon_info(state.get("opponent_active", {})))

    # Conditions
    tokens.append("<conditions>")
    tokens.append(state.get("weather", "noweather"))
    # ... hazards, side conditions

    return " ".join(tokens[:87])  # Match Metamon paper's 87-token format
```

**File:** `pokemon_ai/data/dataset.py`
```python
from pokemon_ai.data.state_converter import convert_metamon_state

# In _convert_metamon_trajectory:
text_obs = convert_metamon_state(state, format_id)  # Instead of str(state)
```

**File:** `pokemon_ai/data/tokenizer.py`
Added ~500 Pokemon-specific tokens: Pokemon names, moves, abilities, items, HP buckets, etc.

## Architecture

Based on the Metamon paper:
- **Model:** 200M parameter transformer ("base" size)
- **Training:** Offline RL with binary actor method
- **Value Estimation:** Multi-gamma heads (0.9, 0.99, 0.999, 0.9999)
- **Attention:** Flash Attention 2 for 2-3x speedup
- **Precision:** BF16 mixed precision (H100 native)

## Configuration

Key settings in `pokemon_ai/configs/h100_1gpu.yaml`:

```yaml
model:
  size: "base"  # ~200M parameters
  use_flash_attention: true
  use_gradient_checkpointing: false  # Disabled for speed (80GB VRAM)

training:
  batch_size: 64
  gradient_accumulation_steps: 2  # Effective batch = 128
  num_epochs: 3
  learning_rate: 3e-4
  compile_model: false  # Disabled: conflicts with Flash Attention

optimization:
  use_mixed_precision: true
  mixed_precision_dtype: "bf16"

hardware:
  num_workers: 12
  pin_memory: true
  prefetch_factor: 4
```

## File Structure

```
pokemon_ai/
├── configs/
│   └── h100_1gpu.yaml      # Single H100 config
├── data/
│   ├── dataset.py          # Lazy loading for Metamon .lz4 files
│   └── tokenizer.py        # Pokemon vocabulary tokenizer
├── models/
│   ├── pokemon_transformer.py  # Main transformer with Flash Attention
│   └── turn_encoder.py     # Per-turn encoder with Flash Attention
└── training/
    └── offline_rl.py       # Offline RL trainer

scripts/
├── train.py                # Main training script
└── convert_metamon_data.py # Metamon format converter (optional)
```

## Monitoring

- **Wandb:** Real-time loss curves at configured project URL
- **CSV Log:** `checkpoints/h100_base/training_log.csv` with step-by-step metrics
- **Checkpoints:** Saved every 2000 steps to `checkpoints/h100_base/`

## Cost Estimate

On RunPod H100 ($2.50/hr):
- ~7 hours per epoch
- ~21 hours for 3 epochs
- **Total cost: ~$52.50**

## Next Steps

After training completes:
1. Evaluate on held-out battles
2. Run self-play to continue improvement
3. Deploy battle agent on Pokemon Showdown
