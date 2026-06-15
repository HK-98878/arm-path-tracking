"""Closed periodic B-spline path for generalised training.

Generated randomly per episode from control points within a workspace sphere.
Optionally allows some points to slightly exceed the workspace boundary to
train robustness to near-unreachable positions.

Arc-length parameterisation follows the same numerical pattern as Figure8Path.
Orientation uses Rotation Minimizing Frame (RMF) integration.
"""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation

from .base_path import Path
from .rmf import RMFIntegrator
from ..utils.kinematics import rotvec_to_quat, quat_multiply


class BSplinePath(Path):
    """Closed periodic cubic B-spline path.

    Randomly generated each episode. The path is closed (position at
    s=0 equals position at s=total_length). Orientation is computed via
    RMF integration, which avoids discontinuities except at the single
    wrap-around point (one per lap — acceptable for training).

    Arc-length parameterisation: numerical, same pattern as Figure8Path.
    """

    # Fixed z-down orientation: 180° about Y (same as CirclePath fixed mode)
    _FIXED_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

    def __init__(
        self,
        control_points: np.ndarray,
        speed: float,
        n_arc_samples: int = 1000,
        orientation_modes: list = None,
        rock_amplitude: float = 0.175,
        n_oscillations: int = 2,
        closed: bool = False,
    ):
        """Initialise B-spline path.

        Args:
            control_points: (N, 3) control points; N >= 4 for cubic spline
            speed: Constant path speed (m/s); negative for reverse direction
            n_arc_samples: Resolution of arc-length and RMF lookup tables
            orientation_modes: List of allowed modes, e.g. ['fixed'], ['rmf'],
                ['rock_x'], ['rock_y'], ['random_fixed'], or any combination.
                Defaults to ['fixed'] (z-down, same as circle/figure8).
            rock_amplitude: Max tilt angle in radians for rock/random_fixed modes (~0.175 = 10°)
            n_oscillations: Number of rock cycles per lap for rock modes
            closed: If True, fit a closed periodic spline (loop). If False (default),
                fit an open spline — tracking runs from s=0 to s=total_length.
        """
        super().__init__()

        self.speed = speed
        self.closed = closed
        self._n_arc_samples = n_arc_samples
        self._orientation_modes = orientation_modes if orientation_modes else ['fixed']
        self._orientation_mode = self._orientation_modes[0]
        self._rock_amplitude = rock_amplitude
        self._n_oscillations = n_oscillations
        self._random_fixed_quat = None

        self._fit_spline(np.array(control_points, dtype=np.float64))
        self._build_arc_length_table()
        self._build_rmf_table()

    def _fit_spline(self, pts: np.ndarray):
        """Fit cubic B-spline to control points (open or closed)."""
        self._tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], k=3, per=self.closed, s=0)

    def _build_arc_length_table(self):
        """Precompute arc-length as a function of spline parameter u ∈ [0, 1].

        Uses the same trapezoidal pattern as Figure8Path._build_arc_length_table.
        u=0 and u=1 map to the same position (periodic closure); integrating
        from 0 to 1 gives the full path length.
        """
        n = self._n_arc_samples
        self._u_table = np.linspace(0.0, 1.0, n)  # includes both 0 and 1

        dx, dy, dz = splev(self._u_table, self._tck, der=1)
        speeds_at_u = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        du = self._u_table[1] - self._u_table[0]
        self._s_table = np.zeros(n)
        self._s_table[1:] = np.cumsum(0.5 * (speeds_at_u[:-1] + speeds_at_u[1:]) * du)

        self.total_length = float(self._s_table[-1])

    def _s_to_u(self, s: float) -> float:
        """Convert arc-length s to spline parameter u via linear interpolation.

        Same pattern as Figure8Path._s_to_t.
        """
        s = float(s) % self.total_length if self.closed else float(np.clip(s, 0.0, self.total_length))
        idx = int(np.searchsorted(self._s_table, s, side='right')) - 1
        idx = max(0, min(idx, len(self._s_table) - 2))

        s0, s1 = self._s_table[idx], self._s_table[idx + 1]
        u0, u1 = self._u_table[idx], self._u_table[idx + 1]

        if s1 - s0 < 1e-12:
            return float(u0)

        alpha = (s - s0) / (s1 - s0)
        return float(u0 + alpha * (u1 - u0))

    def _build_rmf_table(self):
        """Precompute Rotation Minimizing Frame at each arc-length sample.

        RMF is integrated sequentially: each frame is updated from the previous
        using the change in tangent direction. There will be a small discontinuity
        at the wrap-around (one per lap) because the integration is not closed.
        """
        # Compute all tangents at once (vectorised)
        dx, dy, dz = splev(self._u_table, self._tck, der=1)
        tangents = np.stack([dx, dy, dz], axis=1)  # (n, 3)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        tangents = tangents / norms

        # Sequential RMF integration
        integrator = RMFIntegrator(tangents[0])
        quats = np.empty((self._n_arc_samples, 4))
        quats[0] = integrator.orientation
        for i in range(1, self._n_arc_samples):
            quats[i] = integrator.update(tangents[i])

        self._rmf_table = quats

    def _rmf_at_s(self, s: float) -> np.ndarray:
        """Interpolate RMF orientation at arc-length s using SLERP."""
        s = float(s) % self.total_length if self.closed else float(np.clip(s, 0.0, self.total_length))
        idx = int(np.searchsorted(self._s_table, s, side='right')) - 1
        idx = max(0, min(idx, self._n_arc_samples - 2))

        s0, s1 = self._s_table[idx], self._s_table[idx + 1]
        q0 = self._rmf_table[idx]
        q1 = self._rmf_table[idx + 1]

        if s1 - s0 < 1e-12:
            return q0.copy()

        alpha = float((s - s0) / (s1 - s0))
        return self._slerp(q0, q1, alpha)

    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        """Spherical linear interpolation between two unit quaternions (xyzw)."""
        if np.dot(q0, q1) < 0.0:
            q1 = -q1
        dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
        theta = np.arccos(abs(dot))
        if theta < 1e-8:
            result = (1.0 - alpha) * q0 + alpha * q1
            return result / np.linalg.norm(result)
        sin_theta = np.sin(theta)
        return (np.sin((1.0 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / sin_theta

    # ---- Path interface ----

    def position(self, s: float) -> np.ndarray:
        u = self._s_to_u(s)
        x, y, z = splev(u, self._tck)
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

    def velocity(self, s: float) -> np.ndarray:
        """Unit tangent * speed."""
        u = self._s_to_u(s)
        dx, dy, dz = splev(u, self._tck, der=1)
        t = np.array([float(dx), float(dy), float(dz)], dtype=np.float64)
        norm = np.linalg.norm(t)
        if norm < 1e-10:
            t = np.array([1.0, 0.0, 0.0])
        else:
            t /= norm
        return t * self.speed

    def curvature(self, s: float) -> float:
        """κ = ||r' × r''|| / ||r'||³ from spline parametric derivatives."""
        u = self._s_to_u(s)
        dx1, dy1, dz1 = splev(u, self._tck, der=1)
        dx2, dy2, dz2 = splev(u, self._tck, der=2)
        r_prime = np.array([float(dx1), float(dy1), float(dz1)])
        r_double = np.array([float(dx2), float(dy2), float(dz2)])
        cross = np.cross(r_prime, r_double)
        denom = np.linalg.norm(r_prime) ** 3
        if denom < 1e-12:
            return 0.0
        return float(np.linalg.norm(cross) / denom)

    def orientation(self, s: float) -> np.ndarray:
        if self._orientation_mode == 'fixed':
            return self._FIXED_QUAT.copy()
        if self._orientation_mode == 'rmf':
            return self._rmf_at_s(s)
        if self._orientation_mode == 'random_fixed':
            return self._random_fixed_quat.copy() if self._random_fixed_quat is not None else self._FIXED_QUAT.copy()

        # rock_x / rock_y: sinusoidal oscillation about world axis
        phase = 2 * np.pi * self._n_oscillations * s / self.total_length
        theta = self._rock_amplitude * np.sin(phase)
        if self._orientation_mode == 'rock_x':
            axis = np.array([1.0, 0.0, 0.0])
        elif self._orientation_mode == 'rock_y':
            axis = np.array([0.0, 1.0, 0.0])
        else:
            return self._FIXED_QUAT.copy()
        return quat_multiply(rotvec_to_quat(theta * axis), self._FIXED_QUAT)

    def angular_velocity(self, s: float) -> np.ndarray:
        if self._orientation_mode in ('fixed', 'random_fixed'):
            return np.zeros(3)
        if self._orientation_mode == 'rmf':
            eps = self.total_length / self._n_arc_samples
            q0 = self._rmf_at_s(s)
            q1 = self._rmf_at_s(s + eps)
            R0 = Rotation.from_quat(q0)
            R1 = Rotation.from_quat(q1)
            rotvec = (R1 * R0.inv()).as_rotvec()
            return rotvec * (self.speed / eps)

        # rock_x / rock_y
        phase = 2 * np.pi * self._n_oscillations * s / self.total_length
        dtheta_ds = self._rock_amplitude * (2 * np.pi * self._n_oscillations / self.total_length) * np.cos(phase)
        omega_magnitude = dtheta_ds * self.speed
        if self._orientation_mode == 'rock_x':
            return omega_magnitude * np.array([1.0, 0.0, 0.0])
        elif self._orientation_mode == 'rock_y':
            return omega_magnitude * np.array([0.0, 1.0, 0.0])
        return np.zeros(3)

    def reset_orientation_mode(self, rng=None) -> None:
        """Select orientation mode for this episode."""
        if rng is not None:
            self._orientation_mode = rng.choice(self._orientation_modes)
        else:
            self._orientation_mode = self._orientation_modes[0]

        if self._orientation_mode == 'random_fixed':
            theta_x = rng.uniform(-self._rock_amplitude, self._rock_amplitude) if rng is not None else 0.0
            theta_y = rng.uniform(-self._rock_amplitude, self._rock_amplitude) if rng is not None else 0.0
            q_x = rotvec_to_quat(theta_x * np.array([1.0, 0.0, 0.0]))
            q_y = rotvec_to_quat(theta_y * np.array([0.0, 1.0, 0.0]))
            self._random_fixed_quat = quat_multiply(q_y, quat_multiply(q_x, self._FIXED_QUAT))
        else:
            self._random_fixed_quat = None

    # ---- Random factory ----

    @classmethod
    def random(
        cls,
        center: np.ndarray,
        workspace_radius: float,
        speed: float,
        n_control_points: int,
        rng: np.random.Generator,
        min_curvature_radius: float = 0.05,
        workspace_violation_prob: float = 0.0,
        violation_magnitude: float = 0.0,
        n_arc_samples: int = 1000,
        max_retries: int = 20,
        orientation_modes: list = None,
        rock_amplitude: float = 0.175,
        n_oscillations: int = 2,
        closed: bool = False,
    ) -> 'BSplinePath':
        """Generate a random closed B-spline path.

        Args:
            center: (3,) workspace centre position
            workspace_radius: Reachable sphere radius (m)
            speed: Path speed (m/s)
            n_control_points: Number of random control points (>= 4)
            rng: NumPy random generator
            min_curvature_radius: Minimum allowed radius of curvature (m).
                Paths with tighter corners are retried.
            workspace_violation_prob: Per-point probability of extending beyond sphere
            violation_magnitude: Max extra distance beyond sphere (m)
            n_arc_samples: Arc-length table resolution
            max_retries: Max attempts to satisfy curvature constraint

        Returns:
            BSplinePath instance satisfying curvature constraint if possible,
            otherwise the least-curved candidate from all retries.
        """
        center = np.array(center, dtype=np.float64)
        n_control_points = max(4, n_control_points)

        best: 'BSplinePath | None' = None
        best_min_r = 0.0

        for _ in range(max_retries):
            pts = cls._sample_control_points(
                center, workspace_radius, n_control_points, rng,
                workspace_violation_prob, violation_magnitude
            )
            path = cls(pts, speed, n_arc_samples, orientation_modes=orientation_modes,
                       rock_amplitude=rock_amplitude, n_oscillations=n_oscillations, closed=closed)
            min_r = cls._min_curvature_radius(path)

            if min_r > min_curvature_radius:
                return path

            if best is None or min_r > best_min_r:
                best_min_r = min_r
                best = path

        return best  # type: ignore[return-value]

    @staticmethod
    def _sample_control_points(
        center: np.ndarray,
        workspace_radius: float,
        n: int,
        rng: np.random.Generator,
        violation_prob: float,
        violation_mag: float,
    ) -> np.ndarray:
        """Sample n control points uniformly within workspace sphere."""
        pts = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            # Uniform in sphere volume: r = R * U^(1/3)
            radius = workspace_radius * (rng.uniform() ** (1.0 / 3.0))
            if violation_prob > 0 and rng.uniform() < violation_prob:
                radius += violation_mag * rng.uniform()
            pts[i] = center + radius * direction
        return pts

    @staticmethod
    def _min_curvature_radius(path: 'BSplinePath') -> float:
        """Minimum radius of curvature over all arc-length samples (vectorised)."""
        dx1, dy1, dz1 = splev(path._u_table, path._tck, der=1)
        dx2, dy2, dz2 = splev(path._u_table, path._tck, der=2)
        r_prime = np.stack([dx1, dy1, dz1], axis=1)   # (n, 3)
        r_double = np.stack([dx2, dy2, dz2], axis=1)

        cross = np.cross(r_prime, r_double)             # (n, 3)
        cross_norm = np.linalg.norm(cross, axis=1)
        prime_norm3 = np.linalg.norm(r_prime, axis=1) ** 3

        valid = prime_norm3 > 1e-12
        kappa = np.where(valid, cross_norm / np.where(valid, prime_norm3, 1.0), 0.0)
        max_kappa = float(np.max(kappa))

        return 1.0 / max_kappa if max_kappa > 1e-10 else float('inf')

    def __repr__(self) -> str:
        n_pts = len(self._tck[1][0]) if self._tck else 0
        return (
            f"BSplinePath(n_ctrl={n_pts}, speed={self.speed:.3f}m/s, "
            f"total_length={self.total_length:.3f}m)"
        )
