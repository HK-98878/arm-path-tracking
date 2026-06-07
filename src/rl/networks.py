"""Neural network architectures for PPO actor-critic."""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import Optional, Tuple


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
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

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

        Note:
            Uses tanh squashing with proper Jacobian correction for log_prob.
            log_prob(tanh(x)) = log_prob(x) - sum(log(1 - tanh(x)^2))
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)

        if deterministic:
            x = mean  # Pre-tanh value
        else:
            x = dist.sample()  # Pre-tanh sample

        action = torch.tanh(x)
        # Log prob with Jacobian correction for tanh squashing
        # d/dx tanh(x) = 1 - tanh(x)^2, so log|det(Jacobian)| = sum(log(1 - tanh(x)^2))
        log_prob = dist.log_prob(x).sum(dim=-1)
        log_prob = log_prob - torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)

        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log prob and entropy for given actions.

        Args:
            obs: (batch, obs_dim) observations
            actions: (batch, action_dim) tanh-squashed actions to evaluate

        Returns:
            Tuple of (log_prob, entropy)

        Note:
            Actions are tanh-squashed, so we apply atanh to recover pre-tanh
            values, then compute log_prob with Jacobian correction.
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)

        actions_clamped = torch.clamp(actions, -0.9999, 0.9999)
        x = torch.atanh(actions_clamped)

        log_prob = dist.log_prob(x).sum(dim=-1)
        log_prob = log_prob - torch.sum(torch.log(1 - actions.pow(2) + 1e-6), dim=-1)

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


# ---------------------------------------------------------------------------
# LSTM Actor-Critic (separate actor and critic LSTMs)
# ---------------------------------------------------------------------------

_HiddenState = Tuple[torch.Tensor, torch.Tensor]  # (h, c) each (1, B, H)


class LSTMActor(nn.Module):
    """LSTM-based policy network. Maintains no internal state — callers pass (h, c)."""

    def __init__(self, obs_dim: int, action_dim: int, lstm_hidden_size: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size

        self.input_proj = layer_init(nn.Linear(obs_dim, 128))
        self.lstm = nn.LSTM(128, lstm_hidden_size, num_layers=1, batch_first=True)
        self.mean = layer_init(nn.Linear(lstm_hidden_size, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def init_hidden(self, batch_size: int = 1, device=None) -> _HiddenState:
        h = torch.zeros(1, batch_size, self.lstm_hidden_size)
        c = torch.zeros(1, batch_size, self.lstm_hidden_size)
        if device is not None:
            h, c = h.to(device), c.to(device)
        return h, c

    def _features(
        self, obs: torch.Tensor, hidden: Optional[_HiddenState]
    ) -> Tuple[torch.Tensor, _HiddenState]:
        # obs: (B, obs_dim) → project → (B, 1, 128) → LSTM → (B, H)
        x = torch.relu(self.input_proj(obs)).unsqueeze(1)
        if hidden is None:
            lstm_out, new_hidden = self.lstm(x)
        else:
            lstm_out, new_hidden = self.lstm(x, hidden)
        return lstm_out.squeeze(1), new_hidden

    def get_action(
        self,
        obs: torch.Tensor,
        hidden: Optional[_HiddenState] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, _HiddenState]:
        """Sample action. Returns (action, log_prob, entropy, new_hidden)."""
        features, new_hidden = self._features(obs, hidden)
        mean = self.mean(features)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)

        x = mean if deterministic else dist.sample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x).sum(dim=-1)
        log_prob = log_prob - torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)

        return action, log_prob, entropy, new_hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate log_prob and entropy for stored actions (PPO update).

        Args:
            hidden_h, hidden_c: (B, H) stored per-sample hidden states from buffer.
        """
        h0 = hidden_h.unsqueeze(0)  # (1, B, H)
        c0 = hidden_c.unsqueeze(0)
        features, _ = self._features(obs, (h0, c0))
        mean = self.mean(features)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)

        actions_clamped = torch.clamp(actions, -0.9999, 0.9999)
        x = torch.atanh(actions_clamped)
        log_prob = dist.log_prob(x).sum(dim=-1)
        log_prob = log_prob - torch.sum(torch.log(1 - actions.pow(2) + 1e-6), dim=-1)

        return log_prob, entropy


class LSTMCritic(nn.Module):
    """LSTM-based value network. Independent from LSTMActor."""

    def __init__(self, obs_dim: int, lstm_hidden_size: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.lstm_hidden_size = lstm_hidden_size

        self.input_proj = layer_init(nn.Linear(obs_dim, 128))
        self.lstm = nn.LSTM(128, lstm_hidden_size, num_layers=1, batch_first=True)
        self.value = layer_init(nn.Linear(lstm_hidden_size, 1), std=1.0)

    def init_hidden(self, batch_size: int = 1, device=None) -> _HiddenState:
        h = torch.zeros(1, batch_size, self.lstm_hidden_size)
        c = torch.zeros(1, batch_size, self.lstm_hidden_size)
        if device is not None:
            h, c = h.to(device), c.to(device)
        return h, c

    def _features(
        self, obs: torch.Tensor, hidden: Optional[_HiddenState]
    ) -> Tuple[torch.Tensor, _HiddenState]:
        x = torch.relu(self.input_proj(obs)).unsqueeze(1)
        if hidden is None:
            lstm_out, new_hidden = self.lstm(x)
        else:
            lstm_out, new_hidden = self.lstm(x, hidden)
        return lstm_out.squeeze(1), new_hidden

    def forward(
        self, obs: torch.Tensor, hidden: Optional[_HiddenState] = None
    ) -> Tuple[torch.Tensor, _HiddenState]:
        """Inference forward. Returns (value (B,), new_hidden)."""
        features, new_hidden = self._features(obs, hidden)
        return self.value(features).squeeze(-1), new_hidden

    def evaluate(
        self,
        obs: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor
    ) -> torch.Tensor:
        """Training forward using stored per-sample hidden states. Returns values (B,)."""
        h0 = hidden_h.unsqueeze(0)
        c0 = hidden_c.unsqueeze(0)
        features, _ = self._features(obs, (h0, c0))
        return self.value(features).squeeze(-1)


class LSTMActorCritic(nn.Module):
    """Hybrid: LSTM actor + MLP critic.

    Only the actor carries temporal memory. The MLP critic uses the observation
    directly — position error, velocity, and lookahead already capture state value
    well enough, and keeping the critic stateless avoids stale hidden-state
    contamination of GAE advantage estimates.
    """

    def __init__(self, obs_dim: int, action_dim: int, lstm_hidden_size: int = 256):
        super().__init__()
        self.actor = LSTMActor(obs_dim, action_dim, lstm_hidden_size)
        self.critic = Critic(obs_dim)   # MLP — no hidden state
        self.lstm_hidden_size = lstm_hidden_size

    def init_hidden(self, batch_size: int = 1, device=None) -> _HiddenState:
        """Returns actor (h, c) — only the actor has hidden state."""
        return self.actor.init_hidden(batch_size, device)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        actor_hidden: Optional[_HiddenState],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, _HiddenState]:
        """Inference step. Returns (action, log_prob, entropy, value, new_actor_hidden)."""
        action, log_prob, entropy, new_actor_h = self.actor.get_action(obs, actor_hidden, deterministic)
        value = self.critic(obs)
        return action, log_prob, entropy, value, new_actor_h

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        actor_h: torch.Tensor,
        actor_c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training step. Returns (log_probs, entropy, values)."""
        log_prob, entropy = self.actor.evaluate_actions(obs, actions, actor_h, actor_c)
        value = self.critic(obs)
        return log_prob, entropy, value

    def get_value(self, obs: torch.Tensor, _hidden=None) -> torch.Tensor:
        """Value estimate for GAE bootstrap. Returns (B,) tensor."""
        return self.critic(obs)
