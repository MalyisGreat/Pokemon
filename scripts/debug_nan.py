#!/usr/bin/env python3
"""Debug script to find NaN source"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from pokemon_ai.models import create_pokemon_transformer
from pokemon_ai.models.turn_encoder import TurnEncoderConfig, BatchedTurnEncoder
from pokemon_ai.data import create_dataloader

# Create model
print("Creating model...")
model = create_pokemon_transformer(
    size="small",
    use_flash_attention=False,
    use_gradient_checkpointing=False,
)
model = model.cuda()
model.eval()

# Check model parameters for NaN
print("\nChecking model parameters for NaN:")
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"  NaN in {name}")
    if torch.isinf(param).any():
        print(f"  Inf in {name}")

# Create dataloader
print("Loading data...")
dataloader = create_dataloader(
    data_path="data/replays",
    batch_size=2,
    max_turns=50,
    num_workers=0,
    shuffle=False,
)

# Get one batch
batch = next(iter(dataloader))
batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}

print("\nBatch shapes:")
for k, v in batch.items():
    if torch.is_tensor(v):
        print(f"  {k}: {v.shape}, dtype={v.dtype}, min={v.min().item():.4f}, max={v.max().item():.4f}")

# Check for NaN in inputs
print("\nChecking inputs for NaN/Inf:")
for k, v in batch.items():
    if torch.is_tensor(v) and v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        nan_count = torch.isnan(v).sum().item()
        inf_count = torch.isinf(v).sum().item()
        if nan_count > 0 or inf_count > 0:
            print(f"  {k}: {nan_count} NaN, {inf_count} Inf")

# Test turn encoder separately first
print("\n--- Testing Turn Encoder ---")
with torch.no_grad():
    turn_embeds = model.turn_encoder(
        batch["text_tokens"],
        batch["numerical_features"],
        batch["prev_actions"],
        batch["prev_rewards"],
        batch["turn_mask"],
    )
    print(f"Turn embeds shape: {turn_embeds.shape}")
    print(f"Turn embeds NaN: {torch.isnan(turn_embeds).sum().item()}")
    print(f"Turn embeds Inf: {torch.isinf(turn_embeds).sum().item()}")
    if not torch.isnan(turn_embeds).any():
        print(f"Turn embeds min: {turn_embeds.min().item():.4f}, max: {turn_embeds.max().item():.4f}")

# Test input projection
print("\n--- Testing Input Projection ---")
with torch.no_grad():
    hidden = model.input_projection(turn_embeds)
    print(f"After input_projection NaN: {torch.isnan(hidden).sum().item()}")
    if not torch.isnan(hidden).any():
        print(f"Hidden min: {hidden.min().item():.4f}, max: {hidden.max().item():.4f}")

# Test transformer layers one by one
print("\n--- Testing Transformer Layers ---")
with torch.no_grad():
    hidden_states = model.embed_dropout(hidden)
    batch_size, max_turns = hidden_states.shape[:2]
    device = hidden_states.device

    # Create position IDs
    position_ids = torch.arange(max_turns, device=device).unsqueeze(0).expand(batch_size, -1)

    # Create attention mask (same as in forward)
    turn_mask = batch["turn_mask"]
    causal_mask = torch.triu(
        torch.full((max_turns, max_turns), float("-inf"), device=device),
        diagonal=1,
    )
    padding_mask = (~turn_mask).float() * float("-inf")
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0) + padding_mask

    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Attention mask NaN: {torch.isnan(attention_mask).sum().item()}")
    print(f"Attention mask finite values: {torch.isfinite(attention_mask).sum().item()}")
    print(f"Attention mask -inf count: {(attention_mask == float('-inf')).sum().item()}")

    # Test each layer
    for i, layer in enumerate(model.layers):
        hidden_states, _ = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
        )
        nan_count = torch.isnan(hidden_states).sum().item()
        print(f"Layer {i}: NaN={nan_count}, min={hidden_states.min().item():.4f}, max={hidden_states.max().item():.4f}")
        if nan_count > 0:
            print(f"  First NaN at layer {i}!")
            break

# Forward pass
print("\n--- Running full forward pass ---")
with torch.no_grad():
    outputs = model(
        text_tokens=batch["text_tokens"],
        numerical_features=batch["numerical_features"],
        prev_actions=batch["prev_actions"],
        prev_rewards=batch["prev_rewards"],
        turn_mask=batch["turn_mask"],
        action_mask=batch.get("action_mask"),
    )

print("\nOutput shapes and NaN check:")
for k, v in outputs.items():
    if torch.is_tensor(v):
        nan_count = torch.isnan(v).sum().item()
        inf_count = torch.isinf(v).sum().item()
        print(f"  {k}: {v.shape}, NaN={nan_count}, Inf={inf_count}, min={v.min().item():.4f}, max={v.max().item():.4f}")

# Check actor logits specifically
actor_logits = outputs["actor_logits"]
print(f"\nActor logits sample (first position):")
print(actor_logits[0, 0, :])

# Check if softmax produces valid output
probs = torch.softmax(actor_logits.float(), dim=-1)
print(f"\nSoftmax probs NaN: {torch.isnan(probs).sum().item()}")

# Check returns
for gamma in [0.9, 0.99, 0.999, 0.9999]:
    key = f"returns_{gamma}"
    if key in batch:
        v = batch[key]
        print(f"\n{key}: min={v.min().item():.2f}, max={v.max().item():.2f}, mean={v.mean().item():.2f}")

print("\nDone!")
