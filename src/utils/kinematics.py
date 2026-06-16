"""Kinematics utilities for quaternion operations and frame transformations."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (Hamilton product).

    Args:
        q1: First quaternion [w, x, y, z] or [x, y, z, w]
        q2: Second quaternion [w, x, y, z] or [x, y, z, w]

    Returns:
        Product quaternion (same format as input)

    Note:
        Uses scipy convention [x, y, z, w] internally
    """
    # Detect format (scipy uses [x,y,z,w], some use [w,x,y,z])
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1 * r2).as_quat()


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (conjugate for unit quaternions).

    Args:
        q: Quaternion [x, y, z, w]

    Returns:
        Inverse quaternion
    """
    return R.from_quat(q).inv().as_quat()


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation vector (axis-angle).

    Args:
        q: Quaternion [x, y, z, w]

    Returns:
        Rotation vector (3,) - axis scaled by angle in radians

    Note:
        Rotation vector is smooth near identity (unlike Euler angles)
        and has no gimbal lock singularities.
    """
    return R.from_quat(q).as_rotvec()


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to quaternion.

    Args:
        rotvec: Rotation vector (3,)

    Returns:
        Quaternion [x, y, z, w]
    """
    return R.from_rotvec(rotvec).as_quat()


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.

    Args:
        q: Quaternion [x, y, z, w]

    Returns:
        Rotation matrix (3, 3)
    """
    return R.from_quat(q).as_matrix()


def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion.

    Args:
        mat: Rotation matrix (3, 3)

    Returns:
        Quaternion [x, y, z, w]
    """
    return R.from_matrix(mat).as_quat()


def geodesic_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute geodesic angle between two quaternions.

    Args:
        q1: First quaternion [x, y, z, w]
        q2: Second quaternion [x, y, z, w]

    Returns:
        Angle in radians (always positive, in [0, π])

    Note:
        This is the shortest rotation angle between the two orientations.
    """
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    # Relative rotation
    r_rel = r1.inv() * r2
    # Angle is the magnitude of the rotation vector
    return np.linalg.norm(r_rel.as_rotvec())


def rotation_error_rotvec(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Compute orientation error as rotation vector.

    Args:
        q_current: Current orientation quaternion [x, y, z, w]
        q_target: Target orientation quaternion [x, y, z, w]

    Returns:
        Rotation vector (3,) representing error
        (rotation needed to go from current to target)

    Note:
        This is the preferred representation for orientation error in RL:
        - Smooth near identity
        - No gimbal lock
        - Magnitude proportional to error
    """
    r_current = R.from_quat(q_current)
    r_target = R.from_quat(q_target)
    # World-frame error: R_target * R_current^{-1} gives the rotation axis in world
    # coordinates. This is the angular velocity direction to apply in world frame to
    # rotate the current orientation toward the target.
    # (Note: R_current^{-1} * R_target gives the same angle but in the body/EE frame,
    # which is wrong when the result is used as a world-frame angular command or
    # transformed to EE frame via R_ew @ result.)
    r_error = r_target * r_current.inv()
    return r_error.as_rotvec()


def transform_vector_to_frame(vec_world: np.ndarray, R_world_to_frame: np.ndarray) -> np.ndarray:
    """Transform vector from world frame to another frame.

    Args:
        vec_world: Vector in world frame (3,)
        R_world_to_frame: Rotation matrix from world to target frame (3, 3)

    Returns:
        Vector in target frame (3,)
    """
    return R_world_to_frame @ vec_world


def transform_twist_to_world(twist_ee: np.ndarray, R_ee_to_world: np.ndarray) -> np.ndarray:
    """Transform twist (linear + angular velocity) from EE frame to world frame.

    Args:
        twist_ee: Twist in EE frame (6,) [linear_vel; angular_vel]
        R_ee_to_world: Rotation matrix from EE to world (3, 3)

    Returns:
        Twist in world frame (6,)

    Note:
        Both linear and angular velocities transform the same way for
        body-fixed frames.
    """
    v_ee = twist_ee[:3]  # Linear velocity
    w_ee = twist_ee[3:]  # Angular velocity

    v_world = R_ee_to_world @ v_ee
    w_world = R_ee_to_world @ w_ee

    return np.concatenate([v_world, w_world])


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3D vector.

    Args:
        v: 3D vector

    Returns:
        3x3 skew-symmetric matrix [v]_×

    Note:
        [v]_× @ u = v × u (cross product)
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length.

    Args:
        q: Quaternion [x, y, z, w]

    Returns:
        Normalized quaternion
    """
    return q / np.linalg.norm(q)
