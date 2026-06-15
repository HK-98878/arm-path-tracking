"""Analytic circle path with arc-length parameterization.

This is the simplest path for Step 1 of the build order:
- Closed-form position, velocity, curvature
- Exact arc-length parameterization
- Easy to verify correctness

Supports orientation variation modes for curriculum learning:
- fixed: Constant downward orientation
- rock_x: Oscillate about world X axis
- rock_y: Oscillate about world Y axis
"""

import numpy as np
from typing import List, Optional
from .base_path import Path
from .rmf import rmf_for_planar_curve
from ..utils.kinematics import rotvec_to_quat, quat_multiply


class CirclePath(Path):
    """Analytic circle path in 3D space.

    The circle lies in a plane defined by a normal vector, with constant
    tangential speed. Arc-length parameterization ensures s directly
    corresponds to angle: θ = s / radius.
    """

    def __init__(
        self,
        radius: float,
        center: np.ndarray,
        speed: float,
        normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
        orientation_modes: Optional[List[str]] = None,
        rock_amplitude: float = 0.175,  # ~10 degrees
        n_oscillations: int = 2
    ):
        """Initialize circle path.

        Args:
            radius: Circle radius (meters)
            center: (3,) circle center in world frame
            speed: Constant tangential speed (m/s)
            normal: (3,) plane normal (default: XY plane, Z-up)
            orientation_modes: List of allowed modes ['fixed', 'rock_x', 'rock_y']
            rock_amplitude: Max tilt angle in radians (~0.175 = 10°)
            n_oscillations: Number of rock cycles per lap
        """
        super().__init__()

        self.radius = radius
        self.center = np.array(center, dtype=np.float64)
        self.speed = speed
        self.normal = np.array(normal, dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)

        # Build orthonormal basis for circle plane
        self._build_frame()

        # Total arc length (circumference)
        self.total_length = 2 * np.pi * radius

        # Orientation variation parameters
        self._orientation_modes = orientation_modes if orientation_modes else ['fixed']
        self._rock_amplitude = rock_amplitude
        self._n_oscillations = n_oscillations
        self._orientation_mode = self._orientation_modes[0]
        self._random_fixed_quat = None

    def _build_frame(self):
        """Build orthonormal basis {u, v, normal} for circle plane.

        The circle is parameterized as:
        p(θ) = center + radius * (cos(θ) * u + sin(θ) * v)
        """
        # Choose arbitrary vector not parallel to normal
        if abs(self.normal[2]) < 0.9:
            arbitrary = np.array([0.0, 0.0, 1.0])
        else:
            arbitrary = np.array([1.0, 0.0, 0.0])

        # u = perpendicular to normal (first basis vector in plane)
        self.u = np.cross(self.normal, arbitrary)
        self.u = self.u / np.linalg.norm(self.u)

        # v = perpendicular to both normal and u (second basis vector)
        self.v = np.cross(self.normal, self.u)
        self.v = self.v / np.linalg.norm(self.v)

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
        theta = s / self.radius
        p_local = self.radius * (np.cos(theta) * self.u + np.sin(theta) * self.v)
        return self.center + p_local

    def velocity(self, s: float) -> np.ndarray:
        """Velocity at arc length s. Derivative, scaled by speed

        Args:
            s: Arc length

        Returns:
            velocity: (3,) velocity in world frame (tangent * speed)
        """
        theta = s / self.radius
        tangent = -np.sin(theta) * self.u + np.cos(theta) * self.v

        return self.speed * tangent

    def orientation(self, s: float) -> np.ndarray:
        """Orientation at arc length s, possibly rocking about an axis.

        Args:
            s: Arc length (meters)

        Returns:
            quaternion: (4,) orientation [x, y, z, w]

        Note:
            Base orientation is 180° about Y-axis (EE Z points down).
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

    def curvature(self, s: float) -> float:
        """Curvature at arc length s.

        Args:
            s: Arc length

        Returns:
            κ: Curvature (1/meters)
        """
        return 1.0 / self.radius

    def angular_velocity(self, s: float) -> np.ndarray:
        """Angular velocity at arc length s.

        Args:
            s: Arc length (meters)

        Returns:
            omega: (3,) angular velocity in world frame (rad/s)

        Note:
            For fixed mode, returns zero.
            For rocking modes, returns d(theta)/dt * axis where:
            - d(theta)/dt = d(theta)/ds * ds/dt
            - d(theta)/ds = A * (2*pi*n/L) * cos(phase)
            - ds/dt = speed
        """
        if self._orientation_mode in ('fixed', 'random_fixed'):
            return np.zeros(3)

        # d(theta)/ds = A * (2*pi*n/L) * cos(phase)
        phase = 2 * np.pi * self._n_oscillations * s / self.total_length
        dtheta_ds = self._rock_amplitude * (2 * np.pi * self._n_oscillations / self.total_length) * np.cos(phase)

        # omega = d(theta)/dt = d(theta)/ds * ds/dt
        # Signed speed handles path direction automatically
        omega_magnitude = dtheta_ds * self.speed

        # Determine axis
        if self._orientation_mode == 'rock_x':
            axis = np.array([1.0, 0.0, 0.0])
        elif self._orientation_mode == 'rock_y':
            axis = np.array([0.0, 1.0, 0.0])
        else:
            return np.zeros(3)

        return omega_magnitude * axis

    def tangent(self, s: float) -> np.ndarray:
        """Unit tangent at arc length s.

        Args:
            s: Arc length

        Returns:
            tangent: (3,) unit tangent vector
        """
        theta = s / self.radius
        return -np.sin(theta) * self.u + np.cos(theta) * self.v

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CirclePath(radius={self.radius:.3f}m, "
            f"center={self.center}, speed={self.speed:.3f}m/s, "
            f"normal={self.normal}, mode={self._orientation_mode})"
        )


def create_horizontal_circle(
    radius: float = 0.3,
    height: float = 0.3,
    center_xy: np.ndarray = np.array([0.5, 0.0]),
    speed: float = 0.2
) -> CirclePath:
    """Convenience function to create horizontal circle (XY plane).

    Args:
        radius: Circle radius (meters)
        height: Z-coordinate of circle (meters)
        center_xy: (2,) [x, y] center coordinates
        speed: Tangential speed (m/s)

    Returns:
        CirclePath instance
    """
    center = np.array([center_xy[0], center_xy[1], height])
    normal = np.array([0.0, 0.0, 1.0])  # Z-up

    return CirclePath(radius, center, speed, normal)
