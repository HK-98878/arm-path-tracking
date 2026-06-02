"""Abstract base class for path representation with arc-length parameterization."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class Path(ABC):
    """Abstract path interface with arc-length parameterization.

    All paths provide:
    - position(s): Cartesian position at arc length s
    - velocity(s): Cartesian velocity (tangent * speed)
    - orientation(s): Orientation as quaternion (via RMF or custom)
    - curvature(s): Path curvature (optional, for diagnostics)

    Arc-length parameterization ensures constant-speed motion along
    the geometric path, separating geometry from timing.
    """

    def __init__(self):
        """Initialize path."""
        self.total_length: float = 0.0  # Total arc length

    @abstractmethod
    def position(self, s: float) -> np.ndarray:
        """Get Cartesian position at arc length s.

        Args:
            s: Arc length parameter (meters)

        Returns:
            position: (3,) position in world frame
        """
        pass

    @abstractmethod
    def velocity(self, s: float) -> np.ndarray:
        """Get velocity at arc length s.

        Args:
            s: Arc length parameter

        Returns:
            velocity: (3,) velocity in world frame (tangent * ds/dt)

        Note:
            This includes both path geometry (tangent direction)
            and speed profile (ds/dt).
        """
        pass

    @abstractmethod
    def orientation(self, s: float) -> np.ndarray:
        """Get orientation at arc length s.

        Args:
            s: Arc length parameter

        Returns:
            quaternion: (4,) orientation [x, y, z, w] in world frame

        Note:
            Typically uses Rotation Minimizing Frame (RMF) to avoid
            discontinuities at inflection points (e.g., figure-8 crossover).
        """
        pass

    @abstractmethod
    def angular_velocity(self, s: float) -> np.ndarray:
        """Get angular velocity at arc length s.

        Args:
            s: Arc length parameter

        Returns:
            omega: (3,) angular velocity in world frame (rad/s)

        Note:
            For planar curves, omega = (speed / radius_of_curvature) * plane_normal.
            For general 3D curves, this is the time derivative of orientation
            expressed as an angular velocity vector.
        """
        pass

    def curvature(self, s: float) -> float:
        """Get path curvature at arc length s.

        Args:
            s: Arc length parameter

        Returns:
            κ: Curvature (1/meters)

        Note:
            Optional diagnostic. For circle: κ = 1/r (constant).
        """
        return 0.0  # Default: straight line

    def tangent(self, s: float) -> np.ndarray:
        """Get unit tangent vector at arc length s.

        Args:
            s: Arc length parameter

        Returns:
            tangent: (3,) unit tangent vector

        Note:
            Tangent = velocity / ||velocity||
        """
        vel = self.velocity(s)
        speed = np.linalg.norm(vel)
        if speed < 1e-10:
            return np.array([1.0, 0.0, 0.0])  # Default direction
        return vel / speed

    def get_reference_at_time(
        self,
        t: float,
        s_current: float,
        dt: float
    ) -> dict:
        """Get reference state (position, velocity, orientation) at time t.

        Args:
            t: Current time (seconds)
            s_current: Current arc length
            dt: Time step

        Returns:
            Dictionary with:
            - position: (3,) reference position
            - velocity: (3,) reference velocity
            - orientation: (4,) reference quaternion
            - s: current arc length

        Note:
            Default: open-loop advancement s += ||v|| * dt
            Subclasses can override for closed-loop advancement.
        """
        # Open-loop: advance s by distance traveled
        vel = self.velocity(s_current)
        ds = np.linalg.norm(vel) * dt
        s_next = (s_current + ds) % self.total_length  # Wrap around

        return {
            'position': self.position(s_next),
            'velocity': self.velocity(s_next),
            'orientation': self.orientation(s_next),
            's': s_next
        }

    def lookahead_points(
        self,
        s: float,
        n_points: int,
        ds: float
    ) -> np.ndarray:
        """Get lookahead reference points.

        Args:
            s: Current arc length
            n_points: Number of lookahead points
            ds: Distance between points (meters)

        Returns:
            points: (n_points, 3) array of future positions

        Note:
            Used in observation to give policy anticipation.
        """
        points = []
        for k in range(1, n_points + 1):
            s_k = (s + k * ds) % self.total_length
            points.append(self.position(s_k))

        return np.array(points)

    def sample_trajectory(
        self,
        duration: float,
        dt: float,
        start_s: float = 0.0
    ) -> dict:
        """Sample entire trajectory over duration.

        Args:
            duration: Total time (seconds)
            dt: Time step
            start_s: Starting arc length

        Returns:
            Dictionary with arrays:
            - positions: (T, 3)
            - velocities: (T, 3)
            - orientations: (T, 4)
            - arc_lengths: (T,)
            - times: (T,)
        """
        n_steps = int(duration / dt)
        positions = []
        velocities = []
        orientations = []
        arc_lengths = []
        times = []

        s = start_s
        for i in range(n_steps):
            t = i * dt

            # Store current state (before advancing)
            positions.append(self.position(s))
            velocities.append(self.velocity(s))
            orientations.append(self.orientation(s))
            arc_lengths.append(s)
            times.append(t)

            # Advance arc length
            vel = self.velocity(s)
            ds = np.linalg.norm(vel) * dt
            s = (s + ds) % self.total_length

        return {
            'positions': np.array(positions),
            'velocities': np.array(velocities),
            'orientations': np.array(orientations),
            'arc_lengths': np.array(arc_lengths),
            'times': np.array(times)
        }
