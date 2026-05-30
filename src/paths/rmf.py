"""Rotation Minimizing Frame (RMF) for smooth orientation along paths.

RMF is preferred over Frenet frames for path following because:
- No discontinuities at inflection points (e.g., figure-8 crossover)
- Minimal rotation of the reference frame along the path
- More natural for end-effector orientation tracking
"""

import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R


def compute_rmf_orientation(
    tangent: np.ndarray,
    normal: Optional[np.ndarray] = None,
    previous_orientation: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute RMF orientation from tangent vector.

    Args:
        tangent: (3,) unit tangent vector at current point
        normal: Optional (3,) preferred normal direction (for planar curves)
        previous_orientation: Optional (4,) previous RMF quaternion for continuity

    Returns:
        quaternion: (4,) RMF orientation [x, y, z, w]

    Note:
        For planar curves (circle, figure-8), pass the plane normal.
        For 3D curves, use incremental RMF with previous_orientation.
    """
    tangent = tangent / np.linalg.norm(tangent)  # Ensure unit

    if normal is not None:
        # Planar case: construct RMF from tangent and normal
        normal = normal / np.linalg.norm(normal)

        # Ensure normal is perpendicular to tangent
        normal = normal - np.dot(normal, tangent) * tangent
        normal = normal / np.linalg.norm(normal)

        # Binormal = tangent × normal
        binormal = np.cross(tangent, normal)

        # RMF frame: [tangent, normal, binormal] as columns
        R_mat = np.column_stack([tangent, normal, binormal])

        return R.from_matrix(R_mat).as_quat()

    elif previous_orientation is not None:
        # Incremental RMF for 3D curves
        # This is a simplified version; full RMF needs integration along path

        # Extract previous tangent (first column of rotation matrix)
        R_prev = R.from_quat(previous_orientation).as_matrix()
        tangent_prev = R_prev[:, 0]

        # Rotation from previous tangent to current tangent
        axis = np.cross(tangent_prev, tangent)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-10:
            return previous_orientation

        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(tangent_prev, tangent), -1.0, 1.0))
        R_delta = R.from_rotvec(angle * axis)

        # Apply rotation to previous orientation
        R_new = R_delta * R.from_quat(previous_orientation)

        return R_new.as_quat()

    else: # Initialisation fallback
        z_axis = np.array([0, 0, 1])

        if abs(tangent[2]) < 0.9:
            normal = np.cross(z_axis, tangent)
            normal = normal / np.linalg.norm(normal)
        else:
            x_axis = np.array([1, 0, 0])
            normal = np.cross(x_axis, tangent)
            normal = normal / np.linalg.norm(normal)

        binormal = np.cross(tangent, normal)
        R_mat = np.column_stack([tangent, normal, binormal])

        return R.from_matrix(R_mat).as_quat()


def rmf_for_planar_curve(
    tangent: np.ndarray,
    plane_normal: np.ndarray
) -> np.ndarray:
    """Simplified RMF for planar curves (circle, ellipse, figure-8).

    Args:
        tangent: (3,) tangent direction at point
        plane_normal: (3,) normal to the curve plane

    Returns:
        quaternion: (4,) RMF orientation

    Note:
        For planar curves, RMF is easy: tangent + plane normal define frame.
        This avoids integration and is exact for all planar paths.
    """
    return compute_rmf_orientation(tangent, normal=plane_normal)


class RMFIntegrator:
    """Incremental RMF integration for arbitrary 3D curves.

    For complex 3D paths, RMF must be integrated numerically.
    This class maintains state and updates frame incrementally.
    """

    def __init__(self, initial_tangent: np.ndarray):
        """Initialize RMF integrator.

        Args:
            initial_tangent: (3,) tangent at starting point
        """
        self.orientation = compute_rmf_orientation(initial_tangent)

    def update(self, new_tangent: np.ndarray) -> np.ndarray:
        """Update RMF for new tangent.

        Args:
            new_tangent: (3,) tangent at new point

        Returns:
            quaternion: (4,) updated RMF orientation
        """
        self.orientation = compute_rmf_orientation(
            new_tangent,
            previous_orientation=self.orientation
        )
        return self.orientation

    def reset(self, tangent: np.ndarray):
        """Reset integrator with new starting tangent.

        Args:
            tangent: (3,) new starting tangent
        """
        self.orientation = compute_rmf_orientation(tangent)
