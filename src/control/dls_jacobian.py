"""Damped Least Squares (DLS) Jacobian pseudoinverse with null-space projection.

This module implements the core control layer that converts desired EE twist
to joint position commands, with adaptive damping for singularity robustness
and null-space projection for joint-limit avoidance.
"""

import numpy as np
from typing import Tuple, Optional
from ..utils.manipulability import compute_manipulability, joint_limit_gradient


class DLSController:
    """Damped Least Squares controller for redundancy resolution.

    Converts 6-DOF EE twist commands to n-DOF joint commands using:
    - DLS pseudoinverse with adaptive damping (smooth near singularities)
    - Null-space projection for secondary objectives (joint limit avoidance)
    """

    def __init__(
        self,
        n_joints: int,
        q_limits: Tuple[np.ndarray, np.ndarray],
        k_limit: float = 0.5,
        lambda_min: float = 1e-4,
        lambda_max: float = 0.1,
        manip_threshold: float = 0.05
    ):
        """Initialize DLS controller.

        Args:
            n_joints: Number of robot joints
            q_limits: Tuple of (q_min, q_max) joint limit arrays
            k_limit: Null-space gain for joint limit avoidance
            lambda_min: Minimum damping factor (far from singularities)
            lambda_max: Maximum damping factor (at singularities)
            manip_threshold: Manipulability below which damping increases
        """
        self.n_joints = n_joints
        self.q_min, self.q_max = q_limits
        self.k_limit = k_limit
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.manip_threshold = manip_threshold

    def adaptive_damping(self, jacobian: np.ndarray) -> float:
        """Compute adaptive damping factor λ² based on manipulability.

        Args:
            jacobian: (6, n) Jacobian matrix

        Returns:
            λ²: Damping factor that increases near singularities

        Note:
            - Far from singularities (manip > threshold): λ² ≈ lambda_min
            - Near singularities (manip → 0): λ² → lambda_max
            - Smooth transition prevents discontinuities
        """
        manip = compute_manipulability(jacobian)

        if manip >= self.manip_threshold:
            return self.lambda_min
        else:
            # Quadratic increase as manipulability decreases
            ratio = 1.0 - (manip / self.manip_threshold)
            lambda_sq = self.lambda_min + (self.lambda_max - self.lambda_min) * ratio ** 2
            return lambda_sq

    def dls_pseudoinverse(
        self,
        jacobian: np.ndarray,
        lambda_sq: float
    ) -> np.ndarray:
        """Compute DLS pseudoinverse of Jacobian. More stable than SVD

        Args:
            jacobian: (6, n) Jacobian matrix
            lambda_sq: Damping factor λ²

        Returns:
            J_pinv: (n, 6) DLS pseudoinverse
        """
        JJT = jacobian @ jacobian.T
        JJT_damped = JJT + lambda_sq * np.eye(6)
        J_pinv = jacobian.T @ np.linalg.inv(JJT_damped)

        return J_pinv

    def null_space_projector(
        self,
        jacobian: np.ndarray,
        j_pinv: np.ndarray
    ) -> np.ndarray:
        """Compute null-space projection matrix.

        Args:
            jacobian: (6, n) Jacobian
            j_pinv: (n, 6) Pseudoinverse

        Returns:
            N: (n, n) null-space projector
        """
        return np.eye(self.n_joints) - j_pinv @ jacobian

    def step(
        self,
        dx_ee_world: np.ndarray,
        q_current: np.ndarray,
        jacobian: np.ndarray,
        use_null_space: bool = True
    ) -> np.ndarray:
        """Compute joint position command for desired EE twist, used per control step.

        Args:
            dx_ee_world: (6,) desired EE twist in world frame [v; ω] * dt
            q_current: (n,) current joint positions
            jacobian: (6, n) Jacobian matrix (world frame)
            use_null_space: Whether to use null-space for limit avoidance

        Returns:
            q_cmd: (n,) commanded joint positions
        """
        # Adaptive damping based on manipulability
        lambda_sq = self.adaptive_damping(jacobian)
        j_pinv = self.dls_pseudoinverse(jacobian, lambda_sq)

        # Minimum-norm solution (task-space motion)
        dq_task = j_pinv @ dx_ee_world  # (n,)

        # Null-space term for joint limit avoidance
        if use_null_space:
            N = self.null_space_projector(jacobian, j_pinv)
            z = -self.k_limit * joint_limit_gradient(
                q_current,
                (self.q_min, self.q_max),
                margin=0.1  # Start repulsion 0.1 rad from limits
            )
            dq_null = N @ z
        else:
            dq_null = np.zeros(self.n_joints)

        # Final command: current + task + null-space
        q_cmd = q_current + dq_task + dq_null
        return np.clip(q_cmd, self.q_min, self.q_max)

    def get_info(
        self,
        jacobian: np.ndarray,
        q_current: np.ndarray
    ) -> dict:
        """Get diagnostic information about current configuration.

        Args:
            jacobian: (6, n) Jacobian
            q_current: (n,) joint positions

        Returns:
            Dictionary with diagnostic info
        """
        from ..utils.manipulability import (
            compute_condition_number,
            smallest_singular_value,
            joint_limit_proximity
        )

        manip = compute_manipulability(jacobian)
        lambda_sq = self.adaptive_damping(jacobian)

        return {
            'manipulability': manip,
            'damping_lambda_sq': lambda_sq,
            'condition_number': compute_condition_number(jacobian),
            'smallest_singular_value': smallest_singular_value(jacobian),
            'joint_limit_proximity': joint_limit_proximity(
                q_current,
                (self.q_min, self.q_max)
            ),
            'near_singularity': manip < self.manip_threshold
        }


def compute_jacobian_mujoco(
    model,
    data,
    body_id: int,
    point_offset: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute geometric Jacobian using MuJoCo's built-in functions. Performed in world frame.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_id: Body ID for end-effector
        point_offset: Optional offset from body origin (default: [0,0,0])

    Returns:
        jacobian: (6, nv) Jacobian matrix [J_v; J_ω]
                  J_v is (3, nv) linear velocity Jacobian
                  J_ω is (3, nv) angular velocity Jacobian
    """
    import mujoco

    nv = model.nv  # Number of velocity DOFs
    jacp = np.zeros(3 * nv)  # Linear velocity Jacobian (flattened)
    jacr = np.zeros(3 * nv)  # Angular velocity Jacobian (flattened)

    if point_offset is None:
        point_offset = np.zeros(3)

    # MuJoCo function fills jacp and jacr
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    jacp = jacp.reshape(3, nv)
    jacr = jacr.reshape(3, nv)
    jacobian = np.vstack([jacp, jacr])  # (6, nv)

    return jacobian
