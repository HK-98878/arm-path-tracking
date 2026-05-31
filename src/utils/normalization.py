"""Running normalization for observations and rewards."""

import numpy as np


class RunningMeanStd:
    """Tracks running mean and std of a quantity.

    Uses Welford's online algorithm for numerical stability.
    Supports freezing after warmup to prevent distribution shift.
    """

    def __init__(self, shape=(), epsilon=1e-4):
        """Initialize running statistics.

        Args:
            shape: Shape of the quantity to normalize
            epsilon: Small value to avoid division by zero
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
        self.frozen = False

    def freeze(self):
        """Freeze statistics - no more updates after this."""
        self.frozen = True

    def unfreeze(self):
        """Unfreeze statistics - allow updates again."""
        self.frozen = False

    def update(self, x: np.ndarray):
        """Update statistics with new batch of data.

        Args:
            x: New data (batch_size, *shape) or single sample (*shape)

        Note:
            Does nothing if frozen.
        """
        if self.frozen:
            return

        if x.ndim == 1:
            # Single sample
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1
        else:
            # Batch
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch statistics (more efficient for large batches).

        Args:
            batch_mean: Mean of batch
            batch_var: Variance of batch
            batch_count: Number of samples in batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalize data using current statistics.

        Args:
            x: Data to normalize
            clip: Clip normalized values to [-clip, clip] to prevent extremes

        Returns:
            Normalized data: (x - mean) / sqrt(var + epsilon), clipped
        """
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -clip, clip)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize data (inverse operation).

        Args:
            x: Normalized data

        Returns:
            Original scale data
        """
        return x * np.sqrt(self.var + self.epsilon) + self.mean


class RewardNormalizer:
    """Normalizes rewards by running standard deviation.

    This helps stabilize value function learning by keeping
    returns in a consistent range regardless of reward scale.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        """Initialize reward normalizer.

        Args:
            gamma: Discount factor (used for return estimation)
            epsilon: Small value to avoid division by zero
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = 0.0  # Running discounted return estimate

    def normalize(self, reward: float, done: bool) -> float:
        """Normalize reward and update running statistics.

        Args:
            reward: Raw reward
            done: Episode done flag

        Returns:
            Normalized reward (Python float)
        """
        # Update running return estimate
        self.returns = self.returns * self.gamma + reward

        # Update running std of returns using scalar
        self._update_scalar(self.returns)

        # Reset return estimate on episode end
        if done:
            self.returns = 0.0

        # Normalize by std of returns (not mean - we want to preserve sign)
        std = np.sqrt(self.ret_rms.var + self.epsilon)
        if hasattr(std, 'item'):
            std = std.item()
        return reward / std

    def _update_scalar(self, value: float):
        """Update return statistics with a scalar value."""
        if self.ret_rms.frozen:
            return

        delta = value - self.ret_rms.mean
        self.ret_rms.count += 1
        self.ret_rms.mean += delta / self.ret_rms.count
        # Welford's online variance
        delta2 = value - self.ret_rms.mean
        self.ret_rms.var += (delta * delta2 - self.ret_rms.var) / self.ret_rms.count

    def freeze(self):
        """Freeze the return statistics."""
        self.ret_rms.freeze()

    def unfreeze(self):
        """Unfreeze the return statistics."""
        self.ret_rms.unfreeze()
