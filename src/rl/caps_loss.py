"""CAPS (Continuous Action Policy Smoothness) regularization.

CAPS adds temporal and spatial smoothness terms to the policy loss:
- Temporal: Consecutive actions should be similar
- Spatial: Small state perturbations should yield small action changes

Reference: https://arxiv.org/abs/2012.06644
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class CAPSLoss:
    """CAPS regularization for action smoothness."""

    def __init__(
        self,
        lambda_temporal: float = 0.01,
        lambda_spatial: float = 0.01,
        spatial_noise_std: float = 0.01,
        enabled: bool = True
    ):
        """Initialize CAPS loss.

        Args:
            lambda_temporal: Weight for temporal smoothness term
            lambda_spatial: Weight for spatial smoothness term
            spatial_noise_std: Std of noise for spatial perturbation
            enabled: Whether CAPS is enabled (False for Step 1)
        """
        self.lambda_temporal = lambda_temporal
        self.lambda_spatial = lambda_spatial
        self.spatial_noise_std = spatial_noise_std
        self.enabled = enabled

    def compute_temporal(
        self,
        actor: nn.Module,
        obs_t: torch.Tensor,
        obs_t_next: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute temporal smoothness loss.

        Args:
            actor: Policy network
            obs_t: Observations at time t (batch, obs_dim)
            obs_t_next: Observations at time t+1 (batch, obs_dim)
            valid_mask: Optional mask for valid transitions (excludes episode boundaries)

        Returns:
            L_temporal: Temporal smoothness loss (scalar)

        Note:
            L_t = ||π(s_t) - π(s_{t+1})||²
            Penalizes large action changes between consecutive timesteps
        """
        if not self.enabled or self.lambda_temporal == 0:
            return torch.tensor(0.0, device=obs_t.device)

        # Get actions for consecutive states
        with torch.no_grad():
            action_t = actor.get_action(obs_t, deterministic=True)[0]

        action_t_next = actor.get_action(obs_t_next, deterministic=True)[0]

        # L2 difference per sample
        action_diff = action_t_next - action_t
        per_sample_loss = torch.sum(action_diff ** 2, dim=-1)

        # Apply validity mask (exclude episode boundaries)
        if valid_mask is not None:
            per_sample_loss = per_sample_loss * valid_mask
            # Mean over valid samples only
            n_valid = valid_mask.sum().clamp(min=1.0)
            loss = per_sample_loss.sum() / n_valid
        else:
            loss = torch.mean(per_sample_loss)

        return self.lambda_temporal * loss

    def compute_spatial(
        self,
        actor: nn.Module,
        obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute spatial smoothness loss.

        Args:
            actor: Policy network
            obs: Observations (batch, obs_dim)

        Returns:
            L_spatial: Spatial smoothness loss (scalar)

        Note:
            L_s = ||π(s) - π(s + ε)||² where ε ~ N(0, σ²I)
            Penalizes sensitivity to small state perturbations
        """
        if not self.enabled or self.lambda_spatial == 0:
            return torch.tensor(0.0, device=obs.device)

        # Perturb observations
        noise = torch.randn_like(obs) * self.spatial_noise_std
        obs_perturbed = obs + noise

        # Get actions for original and perturbed states
        action_original = actor.get_action(obs, deterministic=True)[0]

        with torch.no_grad():
            action_perturbed = actor.get_action(obs_perturbed, deterministic=True)[0]

        # L2 difference
        action_diff = action_original - action_perturbed
        loss = torch.mean(torch.sum(action_diff ** 2, dim=-1))

        return self.lambda_spatial * loss

    def compute(
        self,
        actor: nn.Module,
        obs_t: Optional[torch.Tensor] = None,
        obs_t_next: Optional[torch.Tensor] = None,
        obs_spatial: Optional[torch.Tensor] = None,
        valid_temporal_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """Compute total CAPS loss.

        Args:
            actor: Policy network
            obs_t: Optional observations at time t (for temporal)
            obs_t_next: Optional observations at time t+1 (for temporal)
            obs_spatial: Optional observations for spatial term
            valid_temporal_mask: Optional mask for valid temporal transitions

        Returns:
            Dictionary with:
            - caps_loss: Total CAPS loss
            - temporal_loss: Temporal component
            - spatial_loss: Spatial component
        """
        temporal_loss = torch.tensor(0.0)
        spatial_loss = torch.tensor(0.0)

        if self.enabled:
            # Temporal term (requires consecutive observations)
            if obs_t is not None and obs_t_next is not None:
                temporal_loss = self.compute_temporal(
                    actor, obs_t, obs_t_next, valid_temporal_mask
                )

            # Spatial term
            if obs_spatial is not None:
                spatial_loss = self.compute_spatial(actor, obs_spatial)

        caps_loss = temporal_loss + spatial_loss

        return {
            'caps_loss': caps_loss,
            'temporal_loss': temporal_loss,
            'spatial_loss': spatial_loss
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.enabled else "disabled"
        return (
            f"CAPSLoss({status}, λ_t={self.lambda_temporal}, "
            f"λ_s={self.lambda_spatial})"
        )
