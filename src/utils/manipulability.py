"""Manipulability metrics and joint limit proximity computations."""

import numpy as np
from typing import Tuple


def compute_manipulability(jacobian: np.ndarray) -> float:
    """Compute Yoshikawa manipulability measure.

    Args:
        jacobian: (m, n) Jacobian matrix (typically 6xn or 3xn)

    Returns:
        Manipulability scalar (>= 0)
        Higher values = better conditioning, farther from singularities

    Note:
        Manipulability μ = sqrt(det(J @ J^T))
        Near singularities, μ → 0
    """
    JJT = jacobian @ jacobian.T
    det = np.linalg.det(JJT)

    # Ensure numerical stability (det might be slightly negative due to rounding)
    det = max(0.0, det)
    return np.sqrt(det)


def compute_manipulability_linear_angular(jacobian: np.ndarray) -> Tuple[float, float]:
    """Compute separate manipulability for linear and angular parts.

    Args:
        jacobian: (6, n) Jacobian matrix where rows 0-2 are linear, 3-5 are angular

    Returns:
        Tuple of (linear_manipulability, angular_manipulability)

    Note:
        This allows detecting when linear vs angular motions are constrained
        independently. Useful for applying separate damping.
    """
    J_linear = jacobian[:3, :]
    J_angular = jacobian[3:, :]

    manip_linear = compute_manipulability(J_linear)
    manip_angular = compute_manipulability(J_angular)

    return manip_linear, manip_angular


def compute_condition_number(jacobian: np.ndarray) -> float:
    """Compute condition number of Jacobian.

    Args:
        jacobian: (6, n) Jacobian matrix

    Returns:
        Condition number (>= 1)
        Lower values = better conditioning

    Note:
        κ(J) = σ_max / σ_min (ratio of max to min singular values)
        Near singularities, κ → ∞
    """
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values[-1] < 1e-10:
        return 1e10

    return singular_values[0] / singular_values[-1]


def joint_limit_proximity(
    q: np.ndarray,
    q_limits: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Compute proximity to joint limits for each joint.

    Args:
        q: Current joint positions (n,)
        q_limits: Tuple of (q_min, q_max) arrays (n,)

    Returns:
        Proximity array (n,) where each element is in [0, 1]
        0 = at center of range, 1 = at limit
    """
    q_min, q_max = q_limits
    q_mid = (q_min + q_max) / 2
    q_range = q_max - q_min

    proximity = 2 * np.abs(q - q_mid) / q_range
    return np.clip(proximity, 0.0, 1.0)


def joint_limit_gradient(
    q: np.ndarray,
    q_limits: Tuple[np.ndarray, np.ndarray],
    margin: float = 0.1
) -> np.ndarray:
    """Compute gradient of joint limit avoidance potential.

    Args:
        q: Current joint positions (n,)
        q_limits: Tuple of (q_min, q_max) arrays (n,)
        margin: Distance from limit (rad) where repulsion starts

    Returns:
        Gradient (n,) pointing away from limits

    Note:
        Uses quadratic potential within margin:
        U = (q - (q_limit ± margin))^2  if within margin, else 0
        ∇U points away from limits
    """
    q_min, q_max = q_limits
    gradient = np.zeros_like(q)

    for i in range(len(q)):
        # Lower limit repulsion
        if q[i] < q_min[i] + margin:
            gradient[i] = -2 * (q[i] - (q_min[i] + margin))  # Negative → push up

        # Upper limit repulsion
        elif q[i] > q_max[i] - margin:
            gradient[i] = 2 * (q[i] - (q_max[i] - margin))  # Positive → push down

    return gradient


def smallest_singular_value(jacobian: np.ndarray) -> float:
    """Compute smallest singular value of Jacobian.

    Args:
        jacobian: (6, n) Jacobian matrix

    Returns:
        Smallest singular value (>= 0)
    """
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return singular_values[-1]


def is_near_singularity(
    jacobian: np.ndarray,
    threshold: float = 0.05
) -> bool:
    """Check if robot is near a singularity.

    Args:
        jacobian: (6, n) Jacobian matrix
        threshold: Manipulability threshold

    Returns:
        True if near singularity
    """
    manip = compute_manipulability(jacobian)
    return manip < threshold


def joint_velocity_magnitude(q_dot: np.ndarray) -> float:
    """Compute joint velocity magnitude (L2 norm).

    Args:
        q_dot: Joint velocities (n,)

    Returns:
        Scalar magnitude

    Note:
        Used as penalty term in reward to discourage large joint motions
    """
    return np.linalg.norm(q_dot)
