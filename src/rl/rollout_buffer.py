"""Rollout buffer for PPO with GAE (Generalized Advantage Estimation)."""

import numpy as np
import torch
from typing import Generator, Tuple


class RolloutBuffer:
    """Buffer for storing rollouts and computing advantages with GAE."""

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu"
    ):
        """Initialize rollout buffer.

        Args:
            buffer_size: Number of steps to store
            obs_dim: Observation dimension
            action_dim: Action dimension
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device for tensors
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Storage
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # GAE computed values
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add transition to buffer.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Episode done flag
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(self, last_value: float, last_done: bool):
        """Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for last state
            last_done: Whether last state was terminal

        Note:
            GAE-Lambda advantage:
            A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        # Use actual buffer size (might not be full)
        size = self.buffer_size if self.full else self.pos

        last_gae_lam = 0
        for step in reversed(range(size)):
            if step == size - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]

            # TD error
            delta = (
                self.rewards[step]
                + self.gamma * next_value * next_non_terminal
                - self.values[step]
            )

            # GAE
            self.advantages[step] = last_gae_lam = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

        # Returns = advantages + values
        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def get(
        self,
        batch_size: int,
        include_next_obs: bool = False
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """Generate batches for training.

        Args:
            batch_size: Batch size
            include_next_obs: Whether to include next observations for temporal CAPS

        Yields:
            Tuples of (obs, actions, values, log_probs, advantages, returns)
            If include_next_obs: also includes (next_obs, valid_temporal_mask)
        """
        size = self.buffer_size if self.full else self.pos

        # Normalize advantages (important for stability)
        advantages = self.advantages[:size].copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Generate random indices
        indices = np.arange(size)
        np.random.shuffle(indices)

        # For temporal CAPS: precompute next observations and validity mask
        if include_next_obs:
            # Next obs is obs[i+1], but invalid for last step or episode boundaries
            next_observations = np.zeros_like(self.observations[:size])
            valid_temporal = np.ones(size, dtype=np.float32)

            for i in range(size - 1):
                if self.dones[i]:
                    # Episode ended, next obs is from different episode
                    valid_temporal[i] = 0.0
                else:
                    next_observations[i] = self.observations[i + 1]
            # Last step has no next obs
            valid_temporal[size - 1] = 0.0

        # Create batches
        for start_idx in range(0, size, batch_size):
            end_idx = min(start_idx + batch_size, size)
            batch_indices = indices[start_idx:end_idx]

            batch = (
                torch.from_numpy(self.observations[batch_indices]).to(self.device),
                torch.from_numpy(self.actions[batch_indices]).to(self.device),
                torch.from_numpy(self.values[batch_indices]).to(self.device),
                torch.from_numpy(self.log_probs[batch_indices]).to(self.device),
                torch.from_numpy(advantages[batch_indices]).to(self.device),
                torch.from_numpy(self.returns[batch_indices]).to(self.device),
            )

            if include_next_obs:
                batch = batch + (
                    torch.from_numpy(next_observations[batch_indices]).to(self.device),
                    torch.from_numpy(valid_temporal[batch_indices]).to(self.device),
                )

            yield batch

    def reset(self):
        """Reset buffer."""
        self.pos = 0
        self.full = False

    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer_size if self.full else self.pos
