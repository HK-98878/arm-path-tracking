"""Analytic circle path with arc-length parameterization.

This is the simplest path for Step 1 of the build order:
- Closed-form position, velocity, curvature
- Exact arc-length parameterization
- Easy to verify correctness
"""

import numpy as np
from .base_path import Path
from .rmf import rmf_for_planar_curve


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
        normal: np.ndarray = np.array([0.0, 0.0, 1.0])
    ):
        """Initialize circle path.

        Args:
            radius: Circle radius (meters)
            center: (3,) circle center in world frame
            speed: Constant tangential speed (m/s)
            normal: (3,) plane normal (default: XY plane, Z-up)
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
        """RMF orientation at arc length s.

        Args:
            s: Arc length

        Returns:
            quaternion: (4,) RMF orientation [x, y, z, w]

        Note:
            For a planar circle, RMF is straightforward:
            - Tangent: velocity direction
            - Normal: plane normal (constant)
            - Binormal: tangent × normal
        """
        theta = s / self.radius
        tangent = -np.sin(theta) * self.u + np.cos(theta) * self.v

        return rmf_for_planar_curve(tangent, self.normal)

    def curvature(self, s: float) -> float:
        """Curvature at arc length s.

        Args:
            s: Arc length

        Returns:
            κ: Curvature (1/meters)
        """
        return 1.0 / self.radius

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
            f"normal={self.normal})"
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
