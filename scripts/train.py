#!/usr/bin/env python3
"""
Main training script for Pokemon AI

Usage:
    # Single GPU
    python scripts/train.py --config configs/h100_1gpu.yaml

    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus=8 scripts/train.py --config configs/h100_8gpu.yaml

    # With Torchrun
    torchrun --nproc_per_node=8 scripts/train.py --config configs/h100_8gpu.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch

from pokemon_ai.training import OfflineRLTrainer, OfflineRLConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle defaults (simple inheritance)
    if "defaults" in config:
        base_configs = config.pop("defaults")
        merged = {}
        for base in base_configs:
            base_path = Path(config_path).parent / f"{base}.yaml"
            if base_path.exists():
                base_config = load_config(str(base_path))
                merged = deep_merge(merged, base_config)
        config = deep_merge(merged, config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def config_to_offline_rl_config(config: dict) -> OfflineRLConfig:
    """Convert nested config dict to OfflineRLConfig"""
    # Helper to safely get values with type casting
    def get_float(d, key, default):
        val = d.get(key, default)
        return float(val) if val is not None else default

    def get_int(d, key, default):
        val = d.get(key, default)
        return int(val) if val is not None else default

    def get_bool(d, key, default):
        val = d.get(key, default)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes")

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    rl_cfg = config.get("rl", {})
    opt_cfg = config.get("optimization", {})
    ds_cfg = config.get("deepspeed", {})
    log_cfg = config.get("logging", {})
    ckpt_cfg = config.get("checkpoint", {})
    hw_cfg = config.get("hardware", {})

    return OfflineRLConfig(
        # Model
        model_size=model_cfg.get("size", "xl"),

        # Data
        data_path=data_cfg.get("path", "data/replays"),
        max_turns=get_int(data_cfg, "max_turns", 200),
        formats=data_cfg.get("formats"),
        min_elo=get_int(data_cfg, "min_elo", None) if data_cfg.get("min_elo") else None,

        # Training
        batch_size=get_int(training_cfg, "batch_size", 32),
        gradient_accumulation_steps=get_int(training_cfg, "gradient_accumulation_steps", 4),
        num_epochs=get_int(training_cfg, "num_epochs", 10),
        max_steps=get_int(training_cfg, "max_steps", None) if training_cfg.get("max_steps") else None,
        learning_rate=get_float(training_cfg, "learning_rate", 1e-4),
        weight_decay=get_float(training_cfg, "weight_decay", 1e-4),
        max_grad_norm=get_float(training_cfg, "max_grad_norm", 1.0),
        warmup_steps=get_int(training_cfg, "warmup_steps", 1000),
        seed=get_int(training_cfg, "seed", 42),

        # RL
        actor_method=rl_cfg.get("actor_method", "binary"),
        actor_coef=get_float(rl_cfg, "actor_coef", 1.0),
        critic_coef=get_float(rl_cfg, "critic_coef", 10.0),
        entropy_coef=get_float(rl_cfg, "entropy_coef", 0.01),
        maxq_lambda=get_float(rl_cfg, "maxq_lambda", 0.0),
        beta=get_float(rl_cfg, "beta", 0.5),
        gammas=tuple(float(g) for g in rl_cfg.get("gammas", [0.9, 0.99, 0.999, 0.9999])),

        # Optimization
        use_mixed_precision=get_bool(opt_cfg, "use_mixed_precision", True),
        mixed_precision_dtype=opt_cfg.get("mixed_precision_dtype", "bf16"),
        use_gradient_checkpointing=get_bool(model_cfg, "use_gradient_checkpointing", True),
        use_flash_attention=get_bool(model_cfg, "use_flash_attention", True),

        # DeepSpeed
        use_deepspeed=get_bool(ds_cfg, "enabled", True),
        deepspeed_stage=get_int(ds_cfg, "stage", 2),
        offload_optimizer=get_bool(ds_cfg, "offload_optimizer", False),
        offload_param=get_bool(ds_cfg, "offload_param", False),

        # Logging
        log_interval=get_int(log_cfg, "log_interval", 100),
        eval_interval=get_int(log_cfg, "eval_interval", 1000),
        save_interval=get_int(log_cfg, "save_interval", 5000),
        use_wandb=get_bool(log_cfg, "use_wandb", True),
        wandb_project=log_cfg.get("wandb_project", "pokemon-superhuman"),
        wandb_run_name=log_cfg.get("wandb_run_name"),

        # Checkpointing
        output_dir=ckpt_cfg.get("output_dir", "checkpoints"),
        resume_from=ckpt_cfg.get("resume_from"),
        save_total_limit=get_int(ckpt_cfg, "save_total_limit", 5),

        # Hardware
        num_workers=get_int(hw_cfg, "num_workers", 4),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Pokemon AI")
    parser.add_argument(
        "--config",
        type=str,
        default="pokemon_ai/configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by launcher)",
    )

    # Override arguments
    parser.add_argument("--data_path", type=str, help="Override data path")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--num_epochs", type=int, help="Override number of epochs")
    parser.add_argument("--max_steps", type=int, help="Override max steps")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint")
    parser.add_argument("--wandb_run_name", type=str, help="Wandb run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = {}

    # Apply overrides
    if args.data_path:
        config.setdefault("data", {})["path"] = args.data_path
    if args.output_dir:
        config.setdefault("checkpoint", {})["output_dir"] = args.output_dir
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.learning_rate:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.num_epochs:
        config.setdefault("training", {})["num_epochs"] = args.num_epochs
    if args.max_steps:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.resume_from:
        config.setdefault("checkpoint", {})["resume_from"] = args.resume_from
    if args.wandb_run_name:
        config.setdefault("logging", {})["wandb_run_name"] = args.wandb_run_name
    if args.no_wandb:
        config.setdefault("logging", {})["use_wandb"] = False

    # Convert to OfflineRLConfig
    rl_config = config_to_offline_rl_config(config)

    # Print config summary
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("=" * 60)
        print("Pokemon AI Training")
        print("=" * 60)
        print(f"Model size: {rl_config.model_size}")
        print(f"Batch size: {rl_config.batch_size} (per GPU)")
        print(f"Learning rate: {rl_config.learning_rate}")
        print(f"Actor method: {rl_config.actor_method}")
        print(f"DeepSpeed: {rl_config.use_deepspeed} (stage {rl_config.deepspeed_stage})")
        print(f"Mixed precision: {rl_config.mixed_precision_dtype}")
        print("=" * 60)

    # Create trainer and run
    trainer = OfflineRLTrainer(rl_config)
    trainer.train()


if __name__ == "__main__":
    main()
