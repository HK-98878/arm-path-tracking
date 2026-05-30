"""Running normalization for observations and rewards."""

import numpy as np


class RunningMeanStd:
    """Tracks running mean and std of a quantity.

    Uses Welford's online algorithm for numerical stability.
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

    def update(self, x: np.ndarray):
        """Update statistics with new batch of data.

        Args:
            x: New data (batch_size, *shape) or single sample (*shape)
        """
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

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize data using current statistics.

        Args:
            x: Data to normalize

        Returns:
            Normalized data: (x - mean) / sqrt(var + epsilon)
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize data (inverse operation).

        Args:
            x: Normalized data

        Returns:
            Original scale data
        """
        return x * np.sqrt(self.var + self.epsilon) + self.mean
