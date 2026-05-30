"""Neural network architectures for PPO actor-critic."""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import Tuple


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights (orthogonal initialization)."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """Policy network (Gaussian with learned std)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        """Initialize actor network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Tuple of hidden layer sizes
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Feature network
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(layer_init(nn.Linear(prev_size, hidden_size)))
            layers.append(nn.Tanh())
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)

        # Mean head
        self.mean = layer_init(nn.Linear(prev_size, action_dim), std=0.01)

        # Log std (learned parameter, not state-dependent)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: (batch, obs_dim) observations

        Returns:
            Tuple of (mean, std)
        """
        features = self.features(obs)
        mean = self.mean(features)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            obs: (batch, obs_dim) observations
            deterministic: If True, return mean (no sampling)

        Returns:
            Tuple of (action, log_prob, entropy)
        """
        mean, std = self.forward(obs)

        if deterministic:
            action = mean
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        # Clip to [-1, 1] (action space bounds)
        action = torch.tanh(action)

        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log prob and entropy for given actions.

        Args:
            obs: (batch, obs_dim) observations
            actions: (batch, action_dim) actions to evaluate

        Returns:
            Tuple of (log_prob, entropy)
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        # Inverse tanh to get pre-tanh action
        # For simplicity, assume actions are already in correct space
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class Critic(nn.Module):
    """Value network."""

    def __init__(self, obs_dim: int, hidden_sizes=(256, 256)):
        """Initialize critic network.

        Args:
            obs_dim: Observation dimension
            hidden_sizes: Tuple of hidden layer sizes
        """
        super().__init__()

        # Feature network
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(layer_init(nn.Linear(prev_size, hidden_size)))
            layers.append(nn.Tanh())
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)

        # Value head
        self.value = layer_init(nn.Linear(prev_size, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: (batch, obs_dim) observations

        Returns:
            values: (batch,) state values
        """
        features = self.features(obs)
        value = self.value(features).squeeze(-1)
        return value


class ActorCritic(nn.Module):
    """Combined actor-critic network."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes=(256, 256)
    ):
        """Initialize actor-critic.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Tuple of hidden layer sizes (shared or separate)
        """
        super().__init__()

        self.actor = Actor(obs_dim, action_dim, hidden_sizes)
        self.critic = Critic(obs_dim, hidden_sizes)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value.

        Args:
            obs: (batch, obs_dim) observations

        Returns:
            values: (batch,) state values
        """
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value.

        Args:
            obs: (batch, obs_dim) observations
            deterministic: If True, use mean action

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        action, log_prob, entropy = self.actor.get_action(obs, deterministic)
        value = self.critic(obs)
        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions (for PPO update).

        Args:
            obs: (batch, obs_dim) observations
            actions: (batch, action_dim) actions

        Returns:
            Tuple of (log_prob, entropy, value)
        """
        log_prob, entropy = self.actor.evaluate_actions(obs, actions)
        value = self.critic(obs)
        return log_prob, entropy, value
