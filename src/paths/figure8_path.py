"""Figure-8 (Lissajous) path with arc-length parameterization.

Provides shape diversity and mid-path curvature reversal for training:
- Curvature changes sign at the center crossing (inflection points)
- Tests policy's ability to handle varying curvature direction
- Same orientation modes as CirclePath for consistency

Parameterization: x = a*sin(t), y = (a/2)*sin(2t) for t in [0, 2*pi]
"""

import numpy as np
from typing import List, Optional
from .base_path import Path
from ..utils.kinematics import rotvec_to_quat, quat_multiply


class Figure8Path(Path):
    """Figure-8 (Lissajous) path in 3D space.

    The figure-8 lies in a plane defined by a normal vector, with constant
    speed along the arc. Arc-length parameterization is computed numerically.
    """

    def __init__(
        self,
        half_width: float,
        center: np.ndarray,
        speed: float,
        normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
        orientation_modes: Optional[List[str]] = None,
        rock_amplitude: float = 0.175,  # ~10 degrees
        n_oscillations: int = 2,
        n_arc_samples: int = 1000
    ):
        """Initialize figure-8 path.

        Args:
            half_width: The 'a' parameter - half the total width (meters)
            center: (3,) center of figure-8 in world frame
            speed: Constant speed along path (m/s)
            normal: (3,) plane normal (default: XY plane, Z-up)
            orientation_modes: List of allowed modes ['fixed', 'rock_x', 'rock_y']
            rock_amplitude: Max tilt angle in radians (~0.175 = 10 deg)
            n_oscillations: Number of rock cycles per lap
            n_arc_samples: Resolution for arc-length lookup table
        """
        super().__init__()

        self.half_width = half_width
        self.center = np.array(center, dtype=np.float64)
        self.speed = speed
        self.normal = np.array(normal, dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)

        # Build orthonormal basis for figure-8 plane
        self._build_frame()

        # Precompute arc-length lookup table
        self._n_arc_samples = n_arc_samples
        self._build_arc_length_table()

        # Orientation variation parameters
        self._orientation_modes = orientation_modes if orientation_modes else ['fixed']
        self._rock_amplitude = rock_amplitude
        self._random_fixed_quat = None
        self._n_oscillations = n_oscillations
        self._orientation_mode = self._orientation_modes[0]

    def _build_frame(self):
        """Build orthonormal basis {u, v, normal} for figure-8 plane.

        The figure-8 is parameterized as:
        p(t) = center + a*sin(t)*u + (a/2)*sin(2t)*v

        For XY plane (normal=[0,0,1]):
        - u points in +X direction (long axis, ±a extent)
        - v points in +Y direction (short axis, ±a/2 extent)
        """
        # Choose arbitrary vector not parallel to normal
        if abs(self.normal[2]) < 0.9:
            arbitrary = np.array([0.0, 0.0, 1.0])
        else:
            # For Z-up normal, use Y to get u pointing in X
            arbitrary = np.array([0.0, 1.0, 0.0])

        # u = first basis vector in plane (long axis of figure-8)
        self.u = np.cross(self.normal, arbitrary)
        self.u = self.u / np.linalg.norm(self.u)

        # v = second basis vector in plane (short axis of figure-8)
        self.v = np.cross(self.normal, self.u)
        self.v = self.v / np.linalg.norm(self.v)

    def _build_arc_length_table(self):
        """Precompute arc-length as function of parameter t.

        Uses trapezoidal integration of ||dp/dt|| over [0, 2*pi].
        Stores cumulative arc-length s(t) for later interpolation.
        """
        n = self._n_arc_samples
        self._t_table = np.linspace(0, 2 * np.pi, n)

        # Compute ||dp/dt|| at each sample point
        # dp/dt = a*cos(t)*u + a*cos(2t)*v
        a = self.half_width
        speeds = np.zeros(n)
        for i, t in enumerate(self._t_table):
            dx_dt = a * np.cos(t)
            dy_dt = a * np.cos(2 * t)
            speeds[i] = np.sqrt(dx_dt**2 + dy_dt**2)

        # Cumulative arc length via trapezoidal rule
        dt = self._t_table[1] - self._t_table[0]
        self._s_table = np.zeros(n)
        for i in range(1, n):
            self._s_table[i] = self._s_table[i-1] + 0.5 * (speeds[i-1] + speeds[i]) * dt

        # Total arc length
        self.total_length = self._s_table[-1]

    def _s_to_t(self, s: float) -> float:
        """Convert arc-length s to parameter t via interpolation.

        Args:
            s: Arc length (wrapped to [0, total_length))

        Returns:
            t: Parameter in [0, 2*pi)
        """
        # Wrap s to valid range
        s = s % self.total_length

        # Linear interpolation on inverted table
        # Find index where s_table[i] <= s < s_table[i+1]
        idx = np.searchsorted(self._s_table, s, side='right') - 1
        idx = np.clip(idx, 0, len(self._s_table) - 2)

        # Interpolate
        s0, s1 = self._s_table[idx], self._s_table[idx + 1]
        t0, t1 = self._t_table[idx], self._t_table[idx + 1]

        if s1 - s0 < 1e-12:
            return t0

        alpha = (s - s0) / (s1 - s0)
        return t0 + alpha * (t1 - t0)

    def _position_at_t(self, t: float) -> np.ndarray:
        """Get position at parameter t (internal use).

        Args:
            t: Parameter in [0, 2*pi]

        Returns:
            position: (3,) position in world frame
        """
        a = self.half_width
        x_local = a * np.sin(t)
        y_local = (a / 2) * np.sin(2 * t)
        return self.center + x_local * self.u + y_local * self.v

    def _velocity_at_t(self, t: float) -> np.ndarray:
        """Get velocity dp/dt at parameter t (internal use).

        Args:
            t: Parameter

        Returns:
            dp_dt: (3,) velocity in world frame (not scaled by speed yet)
        """
        a = self.half_width
        dx_dt = a * np.cos(t)
        dy_dt = a * np.cos(2 * t)
        return dx_dt * self.u + dy_dt * self.v

    @property
    def radius(self) -> float:
        """Characteristic size (alias for half_width, for compatibility with CirclePath)."""
        return self.half_width

    def reset_orientation_mode(self, rng: np.random.Generator = None):
        """Select a random orientation mode for this episode."""
        if rng is None:
            rng = np.random.default_rng()
        self._orientation_mode = rng.choice(self._orientation_modes)
        if self._orientation_mode == 'random_fixed':
            q_base = np.array([0.0, 1.0, 0.0, 0.0])
            theta_x = rng.uniform(-self._rock_amplitude, self._rock_amplitude)
            theta_y = rng.uniform(-self._rock_amplitude, self._rock_amplitude)
            q_x = rotvec_to_quat(theta_x * np.array([1.0, 0.0, 0.0]))
            q_y = rotvec_to_quat(theta_y * np.array([0.0, 1.0, 0.0]))
            self._random_fixed_quat = quat_multiply(q_y, quat_multiply(q_x, q_base))
        else:
            self._random_fixed_quat = None

    def position(self, s: float) -> np.ndarray:
        """Position at arc length s.

        Args:
            s: Arc length (meters)

        Returns:
            position: (3,) position in world frame
        """
        t = self._s_to_t(s)
        return self._position_at_t(t)

    def velocity(self, s: float) -> np.ndarray:
        """Velocity at arc length s.

        Args:
            s: Arc length

        Returns:
            velocity: (3,) velocity in world frame (tangent * speed)
        """
        t = self._s_to_t(s)
        dp_dt = self._velocity_at_t(t)

        # Normalize to get tangent, then scale by speed
        dp_dt_norm = np.linalg.norm(dp_dt)
        if dp_dt_norm < 1e-10:
            # At inflection point, use numerical derivative
            eps = 1e-6
            p1 = self._position_at_t(t + eps)
            p0 = self._position_at_t(t - eps)
            tangent = (p1 - p0) / (2 * eps)
            tangent = tangent / np.linalg.norm(tangent)
        else:
            tangent = dp_dt / dp_dt_norm

        return self.speed * tangent

    def tangent(self, s: float) -> np.ndarray:
        """Unit tangent at arc length s.

        Args:
            s: Arc length

        Returns:
            tangent: (3,) unit tangent vector
        """
        vel = self.velocity(s)
        speed = np.linalg.norm(vel)
        if speed < 1e-10:
            return self.u  # Default direction
        return vel / speed

    def orientation(self, s: float) -> np.ndarray:
        """Orientation at arc length s, possibly rocking about an axis.

        Args:
            s: Arc length (meters)

        Returns:
            quaternion: (4,) orientation [x, y, z, w]

        Note:
            Base orientation is 180 deg about Y-axis (EE Z points down).
            Rocking modes add sinusoidal oscillation about world X or Y.
        """
        # Base orientation: 180 deg about Y (EE Z points down)
        q_base = np.array([0.0, 1.0, 0.0, 0.0])

        if self._orientation_mode == 'fixed':
            return q_base
        if self._orientation_mode == 'random_fixed':
            return self._random_fixed_quat.copy() if self._random_fixed_quat is not None else q_base

        # Compute rock angle: A * sin(2*pi*n*s/L)
        phase = 2 * np.pi * self._n_oscillations * s / self.total_length
        theta = self._rock_amplitude * np.sin(phase)

        if self._orientation_mode == 'rock_x':
            axis = np.array([1.0, 0.0, 0.0])
        elif self._orientation_mode == 'rock_y':
            axis = np.array([0.0, 1.0, 0.0])
        else:
            return q_base

        return quat_multiply(rotvec_to_quat(theta * axis), q_base)

    def angular_velocity(self, s: float) -> np.ndarray:
        """Angular velocity at arc length s.

        Args:
            s: Arc length (meters)

        Returns:
            omega: (3,) angular velocity in world frame (rad/s)

        Note:
            For fixed mode, returns zero.
            For rocking modes, returns d(theta)/dt * axis.
        """
        if self._orientation_mode in ('fixed', 'random_fixed'):
            return np.zeros(3)

        # d(theta)/ds = A * (2*pi*n/L) * cos(phase)
        phase = 2 * np.pi * self._n_oscillations * s / self.total_length
        dtheta_ds = self._rock_amplitude * (2 * np.pi * self._n_oscillations / self.total_length) * np.cos(phase)

        # omega = d(theta)/dt = d(theta)/ds * ds/dt
        omega_magnitude = dtheta_ds * self.speed

        # Determine axis
        if self._orientation_mode == 'rock_x':
            axis = np.array([1.0, 0.0, 0.0])
        elif self._orientation_mode == 'rock_y':
            axis = np.array([0.0, 1.0, 0.0])
        else:
            return np.zeros(3)

        return omega_magnitude * axis

    def curvature(self, s: float) -> float:
        """Curvature at arc length s.

        Args:
            s: Arc length

        Returns:
            kappa: Signed curvature (1/meters)

        Note:
            Curvature changes sign at inflection points (center crossing).
            kappa = (x'*y'' - y'*x'') / (x'^2 + y'^2)^(3/2)
        """
        t = self._s_to_t(s)
        a = self.half_width

        # First derivatives
        dx_dt = a * np.cos(t)
        dy_dt = a * np.cos(2 * t)

        # Second derivatives
        d2x_dt2 = -a * np.sin(t)
        d2y_dt2 = -2 * a * np.sin(2 * t)

        # Curvature formula
        numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
        denominator = (dx_dt**2 + dy_dt**2) ** 1.5

        if abs(denominator) < 1e-12:
            return 0.0

        return numerator / denominator

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Figure8Path(half_width={self.half_width:.3f}m, "
            f"center={self.center}, speed={self.speed:.3f}m/s, "
            f"normal={self.normal}, mode={self._orientation_mode})"
        )


def create_horizontal_figure8(
    half_width: float = 0.15,
    height: float = 0.52,
    center_xy: np.ndarray = np.array([0.32, 0.0]),
    speed: float = 0.1
) -> Figure8Path:
    """Convenience function to create horizontal figure-8 (XY plane).

    Args:
        half_width: Half the total width (meters)
        height: Z-coordinate of figure-8 (meters)
        center_xy: (2,) [x, y] center coordinates
        speed: Speed along path (m/s)

    Returns:
        Figure8Path instance
    """
    center = np.array([center_xy[0], center_xy[1], height])
    normal = np.array([0.0, 0.0, 1.0])  # Z-up

    return Figure8Path(half_width, center, speed, normal)
