"""PPO (Proximal Policy Optimization) implementation with CAPS support.

CleanRL-style: simple, transparent, single-file implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict
import os

from .networks import ActorCritic, LSTMActorCritic
from .rollout_buffer import RolloutBuffer
from .caps_loss import CAPSLoss


class PPO:
    """PPO algorithm with optional CAPS regularization."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_sizes: tuple = (256, 256),
        network_type: str = "mlp",
        lstm_hidden_size: int = 256,
        seq_len: int = 16,
        target_kl: Optional[float] = None,
        device: str = "cpu",
        caps_config: Optional[dict] = None
    ):
        self.device = device
        self.is_lstm = network_type == "lstm"
        self.seq_len = seq_len

        # Network
        if self.is_lstm:
            self.actor_critic = LSTMActorCritic(
                obs_dim, action_dim, lstm_hidden_size
            ).to(device)
            self.actor_hidden, self.critic_hidden = self.actor_critic.init_hidden(device=device)
            # Sequence-boundary hidden states for TBTT: one entry per 16-step chunk
            n_seqs = n_steps // seq_len
            self._seq_actor_h = np.zeros((n_seqs, lstm_hidden_size), dtype=np.float32)
            self._seq_actor_c = np.zeros((n_seqs, lstm_hidden_size), dtype=np.float32)
            self._seq_critic_h = np.zeros((n_seqs, lstm_hidden_size), dtype=np.float32)
            self._seq_critic_c = np.zeros((n_seqs, lstm_hidden_size), dtype=np.float32)
        else:
            self.actor_critic = ActorCritic(
                obs_dim, action_dim, hidden_sizes
            ).to(device)
            self.actor_hidden = None
            self.critic_hidden = None

        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=learning_rate
        )

        # Rollout buffer — TBTT does not store hidden states in buffer
        self.buffer = RolloutBuffer(
            buffer_size=n_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
            lstm_hidden_size=0,
        )

        # CAPS loss
        caps_config = caps_config or {'enabled': False}
        self.caps_loss = CAPSLoss(**caps_config)

        # Hyperparameters
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.target_kl = target_kl
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps

        # Training stats
        self.num_timesteps = 0

    def reset_hidden_state(self):
        """Reset LSTM hidden states to zero (call at episode boundaries)."""
        if self.is_lstm:
            self.actor_hidden, self.critic_hidden = self.actor_critic.init_hidden(device=self.device)

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> tuple:
        """Select action from policy.

        Args:
            obs: Observation
            deterministic: Use deterministic policy (mean)

        Returns:
            Tuple of (action, value, log_prob)
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            if self.is_lstm:
                # Snapshot h at sequence boundaries BEFORE updating hidden state.
                # buffer.pos is where this transition will land; if it's a sequence
                # start, record the hidden state that will initialise that sequence.
                buf_pos = self.buffer.pos
                if buf_pos % self.seq_len == 0:
                    seq_idx = buf_pos // self.seq_len
                    self._seq_actor_h[seq_idx] = self.actor_hidden[0].squeeze().cpu().numpy()
                    self._seq_actor_c[seq_idx] = self.actor_hidden[1].squeeze().cpu().numpy()
                    self._seq_critic_h[seq_idx] = self.critic_hidden[0].squeeze().cpu().numpy()
                    self._seq_critic_c[seq_idx] = self.critic_hidden[1].squeeze().cpu().numpy()
                action, log_prob, _, value, self.actor_hidden, self.critic_hidden = \
                    self.actor_critic.get_action_and_value(
                        obs_tensor, self.actor_hidden, self.critic_hidden, deterministic
                    )
            else:
                action, log_prob, _, value = self.actor_critic.get_action_and_value(
                    obs_tensor, deterministic
                )

        return (
            action.cpu().numpy()[0],
            value.cpu().item(),
            log_prob.cpu().item()
        )

    def log_prob_of_action(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        h_actor=None,
    ) -> float:
        """Compute log_prob for a specific action under the current policy.

        Uses h_actor as the LSTM initial state (should be the hidden state
        *before* the forward pass for this obs, i.e. snapshotted prior to
        select_action). For the MLP case h_actor is ignored.
        """
        from torch.distributions import Normal as _Normal
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            act_t = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
            if self.is_lstm:
                actor = self.actor_critic.actor
                features, _ = actor._features(obs_t, h_actor)
                mean = actor.mean(features)
                std = torch.exp(actor.log_std)
                act_clamped = torch.clamp(act_t, -0.9999, 0.9999)
                x = torch.atanh(act_clamped)
                lp = _Normal(mean, std).log_prob(x).sum(-1)
                lp = lp - torch.sum(torch.log(1 - act_t.pow(2) + 1e-6), dim=-1)
            else:
                lp, _, _ = self.actor_critic.evaluate_actions(obs_t, act_t)
        return lp.cpu().item()

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in buffer.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability
            done: Done flag
        """
        self.buffer.add(obs, action, reward, value, log_prob, done)

    def update(
        self,
        last_obs: np.ndarray,
        last_done: bool
    ) -> Dict[str, float]:
        """Update policy (called after collecting n_steps).

        Args:
            last_obs: Last observation (for bootstrap)
            last_done: Whether last state was terminal

        Returns:
            Dictionary of training metrics
        """
        # Compute last value for GAE
        with torch.no_grad():
            last_obs_tensor = torch.from_numpy(last_obs).float().unsqueeze(0).to(self.device)
            if self.is_lstm:
                last_value = self.actor_critic.get_value(
                    last_obs_tensor, self.critic_hidden
                ).cpu().item()
            else:
                last_value = self.actor_critic.get_value(last_obs_tensor).cpu().item()

        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(last_value, last_done)

        # Training metrics
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'caps_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
        }
        n_batches = 0

        if self.is_lstm:
            # TBTT: recompute hidden states from scratch over seq_len-step sequences.
            # No stored hidden needed — buffer just yields raw (obs, actions, dones, ...) sequences.
            seq_len = self.seq_len
            batch_size_seq = max(1, self.batch_size // seq_len)

            for _epoch in range(self.n_epochs):
                for batch in self.buffer.get_sequences(
                    seq_len, batch_size_seq,
                    self._seq_actor_h, self._seq_actor_c,
                    self._seq_critic_h, self._seq_critic_c,
                ):
                    (obs_seq, actions_seq, dones_seq, old_values_seq,
                     old_log_probs_seq, adv_seq, returns_seq,
                     ah0_b, ac0_b, ch0_b, cc0_b) = batch
                    B, T = obs_seq.shape[:2]

                    # Reconstruct (h, c) tuples in (1, B, H) format for LSTM
                    actor_h0 = (ah0_b.unsqueeze(0), ac0_b.unsqueeze(0))
                    critic_h0 = (ch0_b.unsqueeze(0), cc0_b.unsqueeze(0))

                    # Forward through sequences — gradients flow through T LSTM steps
                    # h0 from rollout gives correct starting context (not zeros)
                    new_log_probs_seq, entropy_seq, new_values_seq = \
                        self.actor_critic.evaluate_sequence(
                            obs_seq, actions_seq, dones_seq, actor_h0, critic_h0
                        )

                    # Flatten (B, T) → (B*T,) for scalar PPO losses
                    new_log_probs = new_log_probs_seq.reshape(-1)
                    entropy = entropy_seq.reshape(-1)
                    new_values = new_values_seq.reshape(-1)
                    old_log_probs = old_log_probs_seq.reshape(-1)
                    old_values = old_values_seq.reshape(-1)
                    advantages = adv_seq.reshape(-1)
                    returns = returns_seq.reshape(-1)

                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    ) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_pred_clipped = old_values + torch.clamp(
                        new_values - old_values, -self.clip_epsilon, self.clip_epsilon
                    )
                    value_loss = 0.5 * torch.max(
                        (new_values - returns) ** 2,
                        (value_pred_clipped - returns) ** 2
                    ).mean()

                    entropy_loss = -entropy.mean()

                    caps_loss = torch.tensor(0.0, device=self.device)
                    if self.caps_loss.enabled:
                        obs_flat = obs_seq.reshape(B * T, -1)
                        caps_dict = self.caps_loss.compute(
                            self.actor_critic.actor,
                            obs_spatial=obs_flat,
                        )
                        caps_loss = caps_dict['caps_loss']

                    loss = (
                        policy_loss
                        + self.vf_coef * value_loss
                        + self.ent_coef * entropy_loss
                        + caps_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - ratio.log()).mean()
                        clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                    metrics['policy_loss'] += policy_loss.item()
                    metrics['value_loss'] += value_loss.item()
                    metrics['entropy_loss'] += entropy_loss.item()
                    metrics['caps_loss'] += caps_loss.item()
                    metrics['approx_kl'] += approx_kl.item()
                    metrics['clip_fraction'] += clip_frac.item()
                    n_batches += 1

                    if self.target_kl is not None and approx_kl.item() > self.target_kl:
                        break  # stop this epoch early

                if self.target_kl is not None and approx_kl.item() > self.target_kl:
                    break  # stop epoch loop

        else:
            # MLP path — unchanged shuffled-batch training
            include_next_obs = (
                self.caps_loss.enabled and self.caps_loss.lambda_temporal > 0
            )
            for _epoch in range(self.n_epochs):
                for batch in self.buffer.get(self.batch_size, include_next_obs=include_next_obs):
                    if include_next_obs:
                        (
                            obs_batch, actions_batch, old_values_batch,
                            old_log_probs_batch, advantages_batch, returns_batch,
                            next_obs_batch, valid_temporal_mask,
                        ) = batch
                    else:
                        (
                            obs_batch, actions_batch, old_values_batch,
                            old_log_probs_batch, advantages_batch, returns_batch,
                        ) = batch
                        next_obs_batch = None
                        valid_temporal_mask = None

                    new_log_probs, entropy, new_values = \
                        self.actor_critic.evaluate_actions(obs_batch, actions_batch)

                    ratio = torch.exp(new_log_probs - old_log_probs_batch)
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    ) * advantages_batch
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_pred_clipped = old_values_batch + torch.clamp(
                        new_values - old_values_batch, -self.clip_epsilon, self.clip_epsilon
                    )
                    value_loss = 0.5 * torch.max(
                        (new_values - returns_batch) ** 2,
                        (value_pred_clipped - returns_batch) ** 2
                    ).mean()

                    entropy_loss = -entropy.mean()

                    caps_loss = torch.tensor(0.0, device=self.device)
                    if self.caps_loss.enabled:
                        caps_dict = self.caps_loss.compute(
                            self.actor_critic.actor,
                            obs_t=obs_batch if include_next_obs else None,
                            obs_t_next=next_obs_batch,
                            obs_spatial=obs_batch,
                            valid_temporal_mask=valid_temporal_mask,
                        )
                        caps_loss = caps_dict['caps_loss']

                    loss = (
                        policy_loss
                        + self.vf_coef * value_loss
                        + self.ent_coef * entropy_loss
                        + caps_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - ratio.log()).mean()
                        clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                    metrics['policy_loss'] += policy_loss.item()
                    metrics['value_loss'] += value_loss.item()
                    metrics['entropy_loss'] += entropy_loss.item()
                    metrics['caps_loss'] += caps_loss.item()
                    metrics['approx_kl'] += approx_kl.item()
                    metrics['clip_fraction'] += clip_frac.item()
                    n_batches += 1

                    if self.target_kl is not None and approx_kl.item() > self.target_kl:
                        break

                if self.target_kl is not None and approx_kl.item() > self.target_kl:
                    break

        for key in metrics:
            metrics[key] /= max(1, n_batches)

        # Reset buffer
        self.buffer.reset()
        self.num_timesteps += self.n_steps

        return metrics

    def save(self, path: str, obs_rms=None, reward_normalizer=None):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            obs_rms: Optional observation normalization stats
            reward_normalizer: Optional reward normalizer
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            'network_type': 'lstm' if self.is_lstm else 'mlp',
            'lstm_hidden_size': self.buffer.lstm_hidden_size,
        }

        if obs_rms is not None:
            checkpoint['obs_rms_mean'] = obs_rms.mean
            checkpoint['obs_rms_var'] = obs_rms.var
            checkpoint['obs_rms_count'] = obs_rms.count
            checkpoint['obs_rms_frozen'] = obs_rms.frozen

        if reward_normalizer is not None:
            checkpoint['reward_rms_mean'] = reward_normalizer.ret_rms.mean
            checkpoint['reward_rms_var'] = reward_normalizer.ret_rms.var
            checkpoint['reward_rms_count'] = reward_normalizer.ret_rms.count
            checkpoint['reward_rms_frozen'] = reward_normalizer.ret_rms.frozen

        torch.save(checkpoint, path)

    def load(self, path: str, obs_rms=None, reward_normalizer=None):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
            obs_rms: Optional observation normalization stats to restore
            reward_normalizer: Optional reward normalizer to restore
        """
        checkpoint = torch.load(path, map_location=self.device,weights_only=False)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.num_timesteps = checkpoint['num_timesteps']

        if obs_rms is not None and 'obs_rms_mean' in checkpoint:
            obs_rms.mean = checkpoint['obs_rms_mean']
            obs_rms.var = checkpoint['obs_rms_var']
            obs_rms.count = checkpoint['obs_rms_count']
            obs_rms.frozen = checkpoint.get('obs_rms_frozen', False)

        if reward_normalizer is not None and 'reward_rms_mean' in checkpoint:
            reward_normalizer.ret_rms.mean = checkpoint['reward_rms_mean']
            reward_normalizer.ret_rms.var = checkpoint['reward_rms_var']
            reward_normalizer.ret_rms.count = checkpoint['reward_rms_count']
            reward_normalizer.ret_rms.frozen = checkpoint.get('reward_rms_frozen', False)

    def set_learning_rate(self, lr: float):
        """Update learning rate for optimizer.

        Args:
            lr: New learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_entropy_coef(self, ent_coef: float):
        """Update entropy coefficient.

        Args:
            ent_coef: New entropy coefficient
        """
        self.ent_coef = ent_coef

    def get_entropy_coef(self) -> float:
        """Get current entropy coefficient."""
        return self.ent_coef
