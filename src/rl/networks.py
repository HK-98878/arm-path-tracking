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

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        h0: Optional[_HiddenState] = None,
    ) -> Tuple[torch.Tensor, _HiddenState]:
        """Run LSTM through a sequence, resetting hidden state at episode boundaries.

        Args:
            obs_seq:   (B, T, obs_dim)
            dones_seq: (B, T) — done[b,t]=1 means episode ended at step t,
                       so hidden is zeroed before step t+1
            h0:        initial hidden state or None (zeros)

        Returns:
            features: (B, T, H), final_hidden: (h, c)
        """
        B, T, _ = obs_seq.shape
        if h0 is None:
            h = torch.zeros(1, B, self.lstm_hidden_size, device=obs_seq.device)
            c = torch.zeros(1, B, self.lstm_hidden_size, device=obs_seq.device)
        else:
            h, c = h0

        features_list = []
        for t in range(T):
            if t > 0:
                mask = (1.0 - dones_seq[:, t - 1]).view(1, B, 1)
                h = h * mask
                c = c * mask
            x = torch.relu(self.input_proj(obs_seq[:, t])).unsqueeze(1)  # (B, 1, 128)
            lstm_out, (h, c) = self.lstm(x, (h, c))
            features_list.append(lstm_out.squeeze(1))  # (B, H)

        return torch.stack(features_list, dim=1), (h, c)  # (B, T, H)

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        h0: Optional[_HiddenState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TBTT training path. Returns log_probs (B, T) and entropy (B, T)."""
        features, _ = self.forward_sequence(obs_seq, dones_seq, h0=h0)
        B, T, H = features.shape
        features_flat = features.reshape(B * T, H)
        actions_flat = actions_seq.reshape(B * T, self.action_dim)

        mean = self.mean(features_flat)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1)

        actions_clamped = torch.clamp(actions_flat, -0.9999, 0.9999)
        x = torch.atanh(actions_clamped)
        log_prob = dist.log_prob(x).sum(dim=-1)
        log_prob = log_prob - torch.sum(torch.log(1 - actions_flat.pow(2) + 1e-6), dim=-1)

        return log_prob.view(B, T), entropy.view(B, T)


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

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        h0: Optional[_HiddenState] = None,
    ) -> Tuple[torch.Tensor, _HiddenState]:
        """(B, T, obs_dim) → features (B, T, H). Resets hidden at episode boundaries."""
        B, T, _ = obs_seq.shape
        if h0 is None:
            h = torch.zeros(1, B, self.lstm_hidden_size, device=obs_seq.device)
            c = torch.zeros(1, B, self.lstm_hidden_size, device=obs_seq.device)
        else:
            h, c = h0

        features_list = []
        for t in range(T):
            if t > 0:
                mask = (1.0 - dones_seq[:, t - 1]).view(1, B, 1)
                h = h * mask
                c = c * mask
            x = torch.relu(self.input_proj(obs_seq[:, t])).unsqueeze(1)
            lstm_out, (h, c) = self.lstm(x, (h, c))
            features_list.append(lstm_out.squeeze(1))

        return torch.stack(features_list, dim=1), (h, c)

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        h0: Optional[_HiddenState] = None,
    ) -> torch.Tensor:
        """TBTT training path. Returns values (B, T)."""
        features, _ = self.forward_sequence(obs_seq, dones_seq, h0=h0)
        B, T, H = features.shape
        return self.value(features.reshape(B * T, H)).squeeze(-1).view(B, T)


class LSTMActorCritic(nn.Module):
    """Full LSTM actor + LSTM critic with TBTT sequence training."""

    def __init__(self, obs_dim: int, action_dim: int, lstm_hidden_size: int = 256):
        super().__init__()
        self.actor = LSTMActor(obs_dim, action_dim, lstm_hidden_size)
        self.critic = LSTMCritic(obs_dim, lstm_hidden_size)
        self.lstm_hidden_size = lstm_hidden_size

    def init_hidden(
        self, batch_size: int = 1, device=None
    ) -> Tuple[_HiddenState, _HiddenState]:
        """Returns (actor_hidden, critic_hidden) — each is (h, c)."""
        return (
            self.actor.init_hidden(batch_size, device),
            self.critic.init_hidden(batch_size, device),
        )

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        actor_hidden: Optional[_HiddenState],
        critic_hidden: Optional[_HiddenState],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, _HiddenState, _HiddenState]:
        """Inference step (single obs). Returns (action, log_prob, entropy, value, new_actor_h, new_critic_h)."""
        action, log_prob, entropy, new_actor_h = self.actor.get_action(obs, actor_hidden, deterministic)
        value, new_critic_h = self.critic.forward(obs, critic_hidden)
        return action, log_prob, entropy, value, new_actor_h, new_critic_h

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        actor_h0: Optional[_HiddenState] = None,
        critic_h0: Optional[_HiddenState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TBTT training. Returns (log_probs (B,T), entropy (B,T), values (B,T))."""
        log_probs, entropy = self.actor.evaluate_sequence(obs_seq, actions_seq, dones_seq, h0=actor_h0)
        values = self.critic.evaluate_sequence(obs_seq, dones_seq, h0=critic_h0)
        return log_probs, entropy, values

    def get_value(
        self, obs: torch.Tensor, critic_hidden: Optional[_HiddenState] = None
    ) -> torch.Tensor:
        """Value estimate for GAE bootstrap. Returns (B,) tensor."""
        value, _ = self.critic.forward(obs, critic_hidden)
        return value
