"""
Offline RL Trainer

Main training loop for Pokemon AI with:
- DeepSpeed ZeRO for efficient multi-GPU training
- Mixed precision (bf16/fp16)
- Gradient checkpointing
- Wandb logging
- Checkpointing and resumption
"""

import os
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from tqdm import tqdm
import json

from pokemon_ai.models import PokemonTransformer, PokemonTransformerConfig, create_pokemon_transformer
from pokemon_ai.data import create_dataloader, PokemonTokenizer
from pokemon_ai.training.losses import CombinedLoss


@dataclass
class OfflineRLConfig:
    """Configuration for offline RL training"""

    # Model
    model_size: str = "xl"  # "small", "base", "large", "xl", "xxl"
    model_config: Optional[Dict] = None  # Override model config

    # Data
    data_path: str = "data/replays"
    max_turns: int = 200
    formats: Optional[List[str]] = None  # Filter formats
    min_elo: Optional[int] = None

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    num_epochs: int = 10
    max_steps: Optional[int] = None  # Override epochs
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000

    # RL specific
    actor_method: str = "binary"  # "il", "exp", "binary", "binary_maxq"
    actor_coef: float = 1.0
    critic_coef: float = 10.0
    entropy_coef: float = 0.01
    maxq_lambda: float = 0.0
    beta: float = 0.5  # Temperature for exponential weighting
    gammas: Tuple[float, ...] = (0.9, 0.99, 0.999, 0.9999)

    # Optimization
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bf16"  # "bf16" or "fp16"
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True

    # DeepSpeed
    use_deepspeed: bool = True
    deepspeed_stage: int = 2  # ZeRO stage (1, 2, or 3)
    offload_optimizer: bool = False
    offload_param: bool = False

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    use_wandb: bool = True
    wandb_project: str = "pokemon-superhuman"
    wandb_run_name: Optional[str] = None

    # Checkpointing
    output_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    save_total_limit: int = 5

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    seed: int = 42

    # Speed optimizations
    compile_model: bool = False  # torch.compile for PyTorch 2.0+
    preload_to_ram: bool = False  # Load all data to RAM


class OfflineRLTrainer:
    """
    Trainer for offline RL on Pokemon battle data.

    Supports:
    - Single GPU training
    - Multi-GPU with DeepSpeed
    - Mixed precision (bf16/fp16)
    - Various offline RL objectives (IL, AWR, CRR, etc.)
    """

    def __init__(self, config: OfflineRLConfig):
        self.config = config

        # Set seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main_process = self.local_rank == 0

        # Create output directory
        self.output_dir = Path(config.output_dir)
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.loss_fn = None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

    def setup(self):
        """Initialize all training components"""
        self._setup_tokenizer()
        self._setup_model()
        self._setup_data()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_deepspeed_or_accelerate()
        self._setup_logging()

        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

    def _setup_tokenizer(self):
        """Setup tokenizer"""
        vocab_path = Path(self.config.data_path) / "vocab.json"
        if vocab_path.exists():
            self.tokenizer = PokemonTokenizer(vocab_file=str(vocab_path))
        else:
            self.tokenizer = PokemonTokenizer()

    def _setup_model(self):
        """Setup model with speed optimizations"""
        if self.config.model_config:
            config = PokemonTransformerConfig(**self.config.model_config)
            self.model = PokemonTransformer(config)
        else:
            self.model = create_pokemon_transformer(
                size=self.config.model_size,
                use_flash_attention=self.config.use_flash_attention,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                gammas=self.config.gammas,
                num_gammas=len(self.config.gammas),
            )

        num_params = self.model.get_num_params()
        if self.is_main_process:
            print(f"Model size: {num_params / 1e6:.2f}M parameters")

        # torch.compile for faster training (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, "compile"):
            if self.is_main_process:
                print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def _setup_data(self):
        """Setup data loaders with speed optimizations"""
        self.train_dataloader = create_dataloader(
            data_path=self.config.data_path,
            batch_size=self.config.batch_size,
            max_turns=self.config.max_turns,
            num_workers=self.config.num_workers,
            shuffle=True,
            formats=self.config.formats,
            min_elo=self.config.min_elo,
            gammas=self.config.gammas,
            world_size=self.world_size,
            rank=self.local_rank,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            preload_to_ram=self.config.preload_to_ram,
        )

    def _setup_loss(self):
        """Setup loss function"""
        self.loss_fn = CombinedLoss(
            actor_method=self.config.actor_method,
            critic_type="two_hot",
            actor_coef=self.config.actor_coef,
            critic_coef=self.config.critic_coef,
            entropy_coef=self.config.entropy_coef,
            maxq_lambda=self.config.maxq_lambda,
            beta=self.config.beta,
            gammas=self.config.gammas,
        )

    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Calculate total steps
        if self.config.max_steps:
            total_steps = self.config.max_steps
        else:
            steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
            total_steps = steps_per_epoch * self.config.num_epochs

        # Warmup + cosine scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps],
        )

    def _setup_deepspeed_or_accelerate(self):
        """Setup DeepSpeed, DDP, or Accelerate for distributed training"""
        if self.config.use_deepspeed and HAS_DEEPSPEED and torch.cuda.is_available():
            self._setup_deepspeed()
        elif self.world_size > 1 and torch.cuda.is_available():
            # Multi-GPU without DeepSpeed - use DDP
            self._setup_ddp()
        elif HAS_ACCELERATE:
            self._setup_accelerate()
        else:
            # Basic single-GPU setup
            self.model = self.model.to(self.device)
            # Setup mixed precision scaler for non-DDP
            if self.config.use_mixed_precision:
                self.scaler = torch.amp.GradScaler("cuda")
            else:
                self.scaler = None

    def _setup_ddp(self):
        """Setup DistributedDataParallel for multi-GPU training"""
        # Initialize process group if not already done
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # Set device for this process
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

        # Move model to device and wrap with DDP
        self.model = self.model.to(self.device)
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            gradient_as_bucket_view=True,  # Memory optimization
            find_unused_parameters=False,  # Faster if all params used
        )

        # BF16 on H100 doesn't need GradScaler (same exponent range as FP32)
        # Only use scaler for FP16
        if self.config.use_mixed_precision and self.config.mixed_precision_dtype == "fp16":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        if self.is_main_process:
            print(f"DDP initialized with {self.world_size} GPUs")
            print(f"  Per-GPU batch size: {self.config.batch_size}")
            print(f"  Effective batch size: {self.config.batch_size * self.world_size * self.config.gradient_accumulation_steps}")
            print(f"  Mixed precision: {self.config.mixed_precision_dtype} (scaler: {'enabled' if self.scaler else 'disabled'})")

    def _setup_deepspeed(self):
        """Setup DeepSpeed"""
        ds_config = self._get_deepspeed_config()

        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            config=ds_config,
        )

        if self.is_main_process:
            print(f"DeepSpeed initialized with ZeRO stage {self.config.deepspeed_stage}")

    def _get_deepspeed_config(self) -> Dict:
        """Generate DeepSpeed configuration"""
        config = {
            "train_batch_size": self.config.batch_size * self.world_size * self.config.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_clipping": self.config.max_grad_norm,
            "steps_per_print": self.config.log_interval,
            "wall_clock_breakdown": False,
        }

        # Mixed precision
        if self.config.use_mixed_precision:
            if self.config.mixed_precision_dtype == "bf16":
                config["bf16"] = {"enabled": True}
            else:
                config["fp16"] = {
                    "enabled": True,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "initial_scale_power": 16,
                    "hysteresis": 2,
                    "min_loss_scale": 1,
                }

        # ZeRO configuration
        zero_config = {
            "stage": self.config.deepspeed_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        }

        if self.config.deepspeed_stage >= 2:
            zero_config["offload_optimizer"] = {
                "device": "cpu" if self.config.offload_optimizer else "none",
                "pin_memory": True,
            }

        if self.config.deepspeed_stage >= 3:
            zero_config["offload_param"] = {
                "device": "cpu" if self.config.offload_param else "none",
                "pin_memory": True,
            }

        config["zero_optimization"] = zero_config

        # Activation checkpointing
        if self.config.use_gradient_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": True,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": True,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False,
            }

        return config

    def _setup_accelerate(self):
        """Setup Accelerate for simpler distributed training"""
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision_dtype if self.config.use_mixed_precision else "no",
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.scheduler
        )

        self.device = self.accelerator.device

    def _setup_logging(self):
        """Setup wandb logging"""
        if self.config.use_wandb and HAS_WANDB and self.is_main_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "model_size": self.config.model_size,
                    "batch_size": self.config.batch_size * self.world_size,
                    "learning_rate": self.config.learning_rate,
                    "actor_method": self.config.actor_method,
                    "num_params": self.model.get_num_params() if hasattr(self.model, "get_num_params") else 0,
                },
            )

    def train(self):
        """Main training loop"""
        self.setup()

        if self.is_main_process:
            print(f"Starting training for {self.config.num_epochs} epochs")
            print(f"Total steps: {len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps}")

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self._train_epoch()

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final save
        if self.is_main_process:
            self._save_checkpoint("final")

        if self.config.use_wandb and HAS_WANDB and self.is_main_process:
            wandb.finish()

    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        # Set epoch for DistributedSampler (ensures different shuffling each epoch)
        if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        # For tracking improvement
        recent_losses = []
        window_size = 100  # Rolling average window

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            disable=not self.is_main_process,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            loss, metrics = self._training_step(batch)

            # Backward pass
            if self.config.use_deepspeed and HAS_DEEPSPEED and hasattr(self.model, 'backward'):
                self.model.backward(loss)
                self.model.step()
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.global_step += 1
            elif HAS_ACCELERATE and hasattr(self, "accelerator"):
                self.accelerator.backward(loss)
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                # DDP or single GPU path with mixed precision
                loss = loss / self.config.gradient_accumulation_steps
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

            current_loss = loss.item()
            epoch_loss += current_loss
            num_batches += 1

            # Track recent losses for rolling average
            recent_losses.append(current_loss)
            if len(recent_losses) > window_size:
                recent_losses.pop(0)
            rolling_avg = sum(recent_losses) / len(recent_losses)

            # Update best loss
            if rolling_avg < self.best_loss and len(recent_losses) >= window_size:
                self.best_loss = rolling_avg

            # Get individual loss components
            actor_loss = metrics.get("actor_loss", 0)
            critic_loss = metrics.get("critic_loss", 0)
            if torch.is_tensor(actor_loss):
                actor_loss = actor_loss.item()
            if torch.is_tensor(critic_loss):
                critic_loss = critic_loss.item()

            # Always update progress bar with current stats
            elapsed = time.time() - start_time
            # Multiply by world_size to get total samples across all GPUs
            samples_per_sec = (num_batches * self.config.batch_size * self.world_size) / elapsed if elapsed > 0 else 0

            progress_bar.set_postfix({
                "loss": f"{rolling_avg:.4f}",
                "best": f"{self.best_loss:.4f}",
                "actor": f"{actor_loss:.3f}",
                "critic": f"{critic_loss:.3f}",
                "s/s": f"{samples_per_sec:.0f}",
                "gpus": f"{self.world_size}",
            })

            # Detailed logging at intervals
            if self.global_step % self.config.log_interval == 0 and self.is_main_process:
                avg_loss = epoch_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]

                # Log to CSV file
                self._log_to_csv(self.global_step, avg_loss, rolling_avg, actor_loss, critic_loss, lr, samples_per_sec)

                # Print summary every 500 steps
                if self.global_step % 500 == 0:
                    improvement = "improving" if rolling_avg < self.best_loss * 1.1 else "stalled"
                    print(f"\n[Step {self.global_step}] Loss: {rolling_avg:.4f} | Best: {self.best_loss:.4f} | Status: {improvement}")

                if self.config.use_wandb and HAS_WANDB:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/rolling_loss": rolling_avg,
                        "train/best_loss": self.best_loss,
                        "train/learning_rate": lr,
                        "train/epoch": self.epoch,
                        "train/global_step": self.global_step,
                        "train/samples_per_second": samples_per_sec,
                        **{f"train/{k}": v.item() if torch.is_tensor(v) else v for k, v in metrics.items()},
                    }, step=self.global_step)

            # Save checkpoint (avoid saving at step 0)
            if self.global_step > 0 and self.global_step % self.config.save_interval == 0 and self.is_main_process:
                self._save_checkpoint(f"step_{self.global_step}")

            # Check max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # End of epoch summary
        if self.is_main_process:
            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"\n{'='*60}")
            print(f"Epoch {self.epoch} Complete")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Best Loss:    {self.best_loss:.4f}")
            print(f"  Total Steps:  {self.global_step}")
            print(f"{'='*60}\n")

    def _log_to_csv(self, step, avg_loss, rolling_loss, actor_loss, critic_loss, lr, samples_per_sec):
        """Log metrics to CSV file for easy plotting"""
        csv_path = self.output_dir / "training_log.csv"
        write_header = not csv_path.exists()

        with open(csv_path, "a") as f:
            if write_header:
                f.write("step,avg_loss,rolling_loss,actor_loss,critic_loss,lr,samples_per_sec,best_loss\n")
            f.write(f"{step},{avg_loss:.6f},{rolling_loss:.6f},{actor_loss:.6f},{critic_loss:.6f},{lr:.2e},{samples_per_sec:.1f},{self.best_loss:.6f}\n")

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single training step"""
        # Determine autocast dtype
        use_autocast = self.config.use_mixed_precision and not (self.config.use_deepspeed and HAS_DEEPSPEED)
        autocast_dtype = torch.bfloat16 if self.config.mixed_precision_dtype == "bf16" else torch.float16

        # Forward pass through model with optional autocast
        with torch.amp.autocast("cuda", enabled=use_autocast, dtype=autocast_dtype):
            outputs = self.model(
                text_tokens=batch["text_tokens"],
                numerical_features=batch["numerical_features"],
                prev_actions=batch["prev_actions"],
                prev_rewards=batch["prev_rewards"],
                turn_mask=batch["turn_mask"],
                action_mask=batch.get("action_mask"),
            )

            # Compute loss
            loss_dict = self.loss_fn(outputs, batch)
            loss = loss_dict["loss"]

        return loss, loss_dict

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device"""
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_deepspeed and HAS_DEEPSPEED:
            self.model.save_checkpoint(str(checkpoint_dir))
        else:
            # Save model
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            torch.save(model_to_save.state_dict(), checkpoint_dir / "model.pt")

            # Save optimizer and scheduler
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_loss": self.best_loss,
            }, checkpoint_dir / "training_state.pt")

        # Save config
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2, default=str)

        print(f"Saved checkpoint to {checkpoint_dir}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint_dir = Path(checkpoint_path)

        if self.config.use_deepspeed and HAS_DEEPSPEED:
            _, client_state = self.model.load_checkpoint(str(checkpoint_dir))
            self.global_step = client_state.get("global_step", 0)
            self.epoch = client_state.get("epoch", 0)
        else:
            # Load model
            model_path = checkpoint_dir / "model.pt"
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device)
                model_to_load = self.model.module if hasattr(self.model, "module") else self.model
                model_to_load.load_state_dict(state_dict)

            # Load training state
            state_path = checkpoint_dir / "training_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location=self.device)
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.global_step = state["global_step"]
                self.epoch = state["epoch"]
                self.best_loss = state.get("best_loss", float("inf"))

        if self.is_main_process:
            print(f"Resumed from checkpoint {checkpoint_path}")
            print(f"Global step: {self.global_step}, Epoch: {self.epoch}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the most recent"""
        checkpoints = sorted(
            self.output_dir.glob("step_*"),
            key=lambda x: int(x.name.split("_")[1]),
        )

        if len(checkpoints) > self.config.save_total_limit:
            for ckpt in checkpoints[:-self.config.save_total_limit]:
                import shutil
                shutil.rmtree(ckpt)


def train_offline_rl(config: Optional[OfflineRLConfig] = None, **kwargs):
    """
    Main entry point for offline RL training.

    Can be called with a config object or keyword arguments.
    """
    if config is None:
        config = OfflineRLConfig(**kwargs)

    trainer = OfflineRLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    # Default training run
    config = OfflineRLConfig(
        model_size="large",
        data_path="data/replays",
        batch_size=32,
        num_epochs=10,
        learning_rate=1e-4,
        actor_method="binary",
        use_wandb=True,
    )
    train_offline_rl(config)
