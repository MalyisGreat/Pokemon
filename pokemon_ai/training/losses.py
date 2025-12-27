"""
Loss functions for offline RL training

Implements various actor-critic objectives from the paper:
- Behavior Cloning (IL)
- Exponential Advantage Weighting (AWR/AWAC style)
- Binary Advantage Filtering (CRR style)
- MaxQ regularization
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_advantages(
    q_values: torch.Tensor,  # [batch, seq_len]
    action_probs: torch.Tensor,  # [batch, seq_len, num_actions]
    actions: torch.Tensor,  # [batch, seq_len]
) -> torch.Tensor:
    """
    Compute advantage estimates: A(s, a) = Q(s, a) - V(s)

    Where V(s) = E_a~π[Q(s, a)]
    """
    # Get Q-value for taken action
    q_taken = q_values  # Already indexed to taken action

    # Compute V(s) = E[Q(s, a)] under current policy
    # This requires Q-values for all actions, but we only have Q for taken action
    # Approximate with sampled actions or use Monte Carlo estimate
    v_estimate = q_taken  # Simplified: use Q(s, a) as baseline

    # In full implementation, we'd compute V properly:
    # v_estimate = (action_probs * all_q_values).sum(dim=-1)

    advantages = q_taken - v_estimate.detach()
    return advantages


class ActorLoss(nn.Module):
    """
    Actor loss with various offline RL objectives.

    L_actor = -w(h, a) * log π(a|h) - λ * E_a~π[Q(h, a)]

    Where w(h, a) is the advantage weighting function:
    - "il": w = 1 (pure behavior cloning)
    - "exp": w = exp(β * A(h, a)) with clipping
    - "binary": w = 1 if A(h, a) > 0 else 0
    - "binary_maxq": binary + MaxQ regularization
    """

    def __init__(
        self,
        method: str = "exp",  # "il", "exp", "binary", "binary_maxq"
        beta: float = 0.5,  # Temperature for exponential weighting
        exp_clip: Tuple[float, float] = (1e-5, 50.0),  # Clipping for exp weights
        maxq_lambda: float = 0.0,  # Weight for MaxQ regularization
        entropy_coef: float = 0.01,  # Entropy bonus
    ):
        super().__init__()
        self.method = method
        self.beta = beta
        self.exp_clip = exp_clip
        self.maxq_lambda = maxq_lambda
        self.entropy_coef = entropy_coef

    def forward(
        self,
        actor_logits: torch.Tensor,  # [batch, seq_len, num_actions]
        actions: torch.Tensor,  # [batch, seq_len]
        advantages: torch.Tensor,  # [batch, seq_len]
        mask: torch.Tensor,  # [batch, seq_len]
        q_values: Optional[torch.Tensor] = None,  # [batch, seq_len, num_actions] for MaxQ
        action_mask: Optional[torch.Tensor] = None,  # [batch, seq_len, num_actions]
    ) -> Dict[str, torch.Tensor]:
        """Compute actor loss"""
        batch_size, seq_len, num_actions = actor_logits.shape

        # Apply action mask if provided
        if action_mask is not None:
            actor_logits = actor_logits.masked_fill(~action_mask, float("-inf"))

        # Compute log probabilities (use float32 for numerical stability)
        actor_logits_f32 = actor_logits.float()
        log_probs = F.log_softmax(actor_logits_f32, dim=-1)
        probs = F.softmax(actor_logits_f32, dim=-1)

        # Clamp actions to valid range
        actions_clamped = actions.clamp(0, num_actions - 1)

        # Get log prob of taken actions
        action_log_probs = log_probs.gather(-1, actions_clamped.unsqueeze(-1)).squeeze(-1)

        # Compute weights based on method
        if self.method == "il":
            weights = torch.ones_like(advantages)
        elif self.method == "exp":
            weights = torch.exp(self.beta * advantages)
            weights = torch.clamp(weights, self.exp_clip[0], self.exp_clip[1])
        elif self.method in ["binary", "binary_maxq"]:
            weights = (advantages > 0).float()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Weighted negative log likelihood
        bc_loss = -weights * action_log_probs
        bc_loss = (bc_loss * mask).sum() / mask.sum().clamp(min=1)

        # MaxQ regularization: encourage policy to maximize Q
        maxq_loss = torch.tensor(0.0, device=actor_logits.device)
        if self.method == "binary_maxq" and q_values is not None and self.maxq_lambda > 0:
            expected_q = (probs * q_values).sum(dim=-1)
            maxq_loss = -(expected_q * mask).sum() / mask.sum().clamp(min=1)

        # Entropy bonus (avoid NaN from 0 * -inf by clamping probs)
        probs_safe = probs.clamp(min=1e-8)
        log_probs_safe = probs_safe.log()
        entropy = -(probs_safe * log_probs_safe).sum(dim=-1)
        entropy = (entropy * mask).sum() / mask.sum().clamp(min=1)

        # Total loss
        total_loss = bc_loss + self.maxq_lambda * maxq_loss - self.entropy_coef * entropy

        return {
            "actor_loss": total_loss,
            "bc_loss": bc_loss,
            "maxq_loss": maxq_loss,
            "entropy": entropy,
            "mean_weight": (weights * mask).sum() / mask.sum().clamp(min=1),
        }


class CriticLoss(nn.Module):
    """
    Critic loss for Q-value prediction.

    Supports:
    - Standard MSE regression
    - Two-hot classification (more stable)
    - Ensemble with min/mean reduction
    """

    def __init__(
        self,
        loss_type: str = "two_hot",  # "mse" or "two_hot"
        num_bins: int = 128,
        value_range: Tuple[float, float] = (-150.0, 150.0),
        td_lambda: float = 0.95,  # GAE lambda
    ):
        super().__init__()
        self.loss_type = loss_type
        self.num_bins = num_bins
        self.value_range = value_range
        self.td_lambda = td_lambda

        if loss_type == "two_hot":
            self.register_buffer(
                "bin_centers",
                torch.linspace(value_range[0], value_range[1], num_bins)
            )

    def encode_two_hot(self, values: torch.Tensor) -> torch.Tensor:
        """Encode continuous values as two-hot distribution"""
        values = torch.clamp(values, self.value_range[0], self.value_range[1])

        bin_width = (self.value_range[1] - self.value_range[0]) / (self.num_bins - 1)
        normalized = (values - self.value_range[0]) / bin_width

        lower_idx = torch.floor(normalized).long()
        upper_idx = torch.ceil(normalized).long()
        lower_idx = torch.clamp(lower_idx, 0, self.num_bins - 1)
        upper_idx = torch.clamp(upper_idx, 0, self.num_bins - 1)

        upper_weight = normalized - lower_idx.float()
        lower_weight = 1.0 - upper_weight

        two_hot = torch.zeros(*values.shape, self.num_bins, device=values.device)
        two_hot.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        two_hot.scatter_add_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))

        return two_hot

    def decode_two_hot(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits back to continuous values"""
        probs = F.softmax(logits, dim=-1)
        # Ensure bin_centers is on the same device as logits
        bin_centers = self.bin_centers.to(logits.device)
        return (probs * bin_centers).sum(dim=-1)

    def forward(
        self,
        value_logits: torch.Tensor,  # [batch, seq_len, num_bins] or [batch, seq_len]
        target_values: torch.Tensor,  # [batch, seq_len]
        mask: torch.Tensor,  # [batch, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """Compute critic loss"""
        if self.loss_type == "two_hot":
            target_dist = self.encode_two_hot(target_values)
            log_probs = F.log_softmax(value_logits, dim=-1)

            # Cross-entropy loss
            loss = -(target_dist * log_probs).sum(dim=-1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)

            # Compute prediction for logging
            predicted = self.decode_two_hot(value_logits)
            mse = F.mse_loss(predicted * mask, target_values * mask, reduction="sum") / mask.sum().clamp(min=1)
        else:
            loss = F.mse_loss(value_logits * mask, target_values * mask, reduction="sum") / mask.sum().clamp(min=1)
            mse = loss
            predicted = value_logits

        return {
            "critic_loss": loss,
            "critic_mse": mse,
            "predicted_mean": (predicted * mask).sum() / mask.sum().clamp(min=1),
            "target_mean": (target_values * mask).sum() / mask.sum().clamp(min=1),
        }


class TDLoss(nn.Module):
    """
    Temporal Difference loss for value prediction.

    Computes TD targets: y = r + γ * V(s') * (1 - done)
    """

    def __init__(
        self,
        gamma: float = 0.999,
        use_target_network: bool = True,
    ):
        super().__init__()
        self.gamma = gamma
        self.use_target_network = use_target_network

    def compute_td_targets(
        self,
        rewards: torch.Tensor,  # [batch, seq_len]
        values: torch.Tensor,  # [batch, seq_len] - V(s) for all s
        dones: torch.Tensor,  # [batch, seq_len]
        mask: torch.Tensor,  # [batch, seq_len]
    ) -> torch.Tensor:
        """Compute TD(0) targets"""
        batch_size, seq_len = rewards.shape

        # Shift values to get V(s')
        next_values = torch.zeros_like(values)
        next_values[:, :-1] = values[:, 1:]

        # TD target: r + γ * V(s') * (1 - done)
        targets = rewards + self.gamma * next_values * (1 - dones.float())

        return targets

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor,
        lambda_: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        batch_size, seq_len = rewards.shape
        device = rewards.device

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Process each sequence
        for b in range(batch_size):
            gae = 0.0
            seq_mask = mask[b]
            valid_len = seq_mask.sum().int().item()

            for t in reversed(range(valid_len)):
                if t == valid_len - 1:
                    next_value = 0.0
                else:
                    next_value = values[b, t + 1]

                delta = rewards[b, t] + self.gamma * next_value * (1 - dones[b, t].float()) - values[b, t]
                gae = delta + self.gamma * lambda_ * (1 - dones[b, t].float()) * gae

                advantages[b, t] = gae
                returns[b, t] = gae + values[b, t]

        return advantages, returns


class CombinedLoss(nn.Module):
    """
    Combined actor-critic loss for end-to-end training.
    """

    def __init__(
        self,
        actor_method: str = "binary",
        critic_type: str = "two_hot",
        actor_coef: float = 1.0,
        critic_coef: float = 10.0,
        entropy_coef: float = 0.01,
        maxq_lambda: float = 0.0,
        beta: float = 0.5,
        gammas: Tuple[float, ...] = (0.9, 0.99, 0.999, 0.9999),
    ):
        super().__init__()
        self.actor_coef = actor_coef
        self.critic_coef = critic_coef
        self.gammas = gammas

        self.actor_loss = ActorLoss(
            method=actor_method,
            beta=beta,
            maxq_lambda=maxq_lambda,
            entropy_coef=entropy_coef,
        )

        # Use underscores instead of dots for module names (PyTorch requirement)
        def gamma_key(g):
            return f"gamma_{str(g).replace('.', '_')}"

        self.gamma_keys = {g: gamma_key(g) for g in gammas}

        self.critic_losses = nn.ModuleDict({
            gamma_key(gamma): CriticLoss(loss_type=critic_type)
            for gamma in gammas
        })

        self.td_losses = nn.ModuleDict({
            gamma_key(gamma): TDLoss(gamma=gamma)
            for gamma in gammas
        })

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        target_values: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss"""
        actor_logits = model_outputs["actor_logits"]
        mask = batch["turn_mask"]
        actions = batch["actions"]

        # Get value predictions for primary gamma (for advantage computation)
        primary_gamma = self.gammas[-1]  # Use highest gamma
        value_key = f"value_gamma_{primary_gamma}"
        values = model_outputs.get(value_key, torch.zeros_like(mask.float()))

        # Use returns from dataset as targets
        returns_key = f"returns_{primary_gamma}"
        if returns_key in batch:
            target_returns = batch[returns_key]
        else:
            target_returns = batch["rewards"]  # Fallback

        # Compute advantages
        advantages = target_returns - values.detach()

        # Actor loss
        actor_out = self.actor_loss(
            actor_logits=actor_logits,
            actions=actions,
            advantages=advantages,
            mask=mask,
            q_values=None,
            action_mask=batch.get("action_mask"),
        )

        # Critic losses for each gamma
        total_critic_loss = torch.tensor(0.0, device=actor_logits.device)
        critic_metrics = {}

        for gamma in self.gammas:
            value_logits_key = f"value_logits_gamma_{gamma}"
            returns_key = f"returns_{gamma}"

            if value_logits_key in model_outputs and returns_key in batch:
                gamma_key = self.gamma_keys[gamma]
                critic_loss_fn = self.critic_losses[gamma_key]
                critic_out = critic_loss_fn(
                    value_logits=model_outputs[value_logits_key],
                    target_values=batch[returns_key],
                    mask=mask,
                )
                total_critic_loss = total_critic_loss + critic_out["critic_loss"]
                critic_metrics[f"critic_loss_{gamma}"] = critic_out["critic_loss"]

        # Total loss
        total_loss = self.actor_coef * actor_out["actor_loss"] + self.critic_coef * total_critic_loss

        return {
            "loss": total_loss,
            "actor_loss": actor_out["actor_loss"],
            "critic_loss": total_critic_loss,
            "bc_loss": actor_out["bc_loss"],
            "entropy": actor_out["entropy"],
            "mean_advantage": advantages.mean(),
            "mean_weight": actor_out["mean_weight"],
            **critic_metrics,
        }
