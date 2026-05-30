"""Observation builder for end-effector tracking environment.

Constructs the 58-dimensional observation vector (for n=7 joints):
- Position error (EE frame): 3
- Orientation error (rotation vector, EE frame): 3
- Lookahead points (EE frame): 15 (5 points × 3)
- Reference velocity (EE frame): 3
- EE linear velocity (EE frame): 3
- EE angular velocity (EE frame): 3
- Joint positions: 7
- Joint velocities: 7
- Previous action: 6
- Manipulability: 1
- Joint limit proximity: 7

Total: 58 dimensions
"""

import numpy as np
from typing import Tuple
from .robot_state import RobotState
from ..paths.base_path import Path
from ..utils.kinematics import rotation_error_rotvec
from ..utils.manipulability import compute_manipulability, joint_limit_proximity


class ObservationBuilder:
    """Builds observations for the tracking environment."""

    def __init__(
        self,
        n_joints: int,
        joint_limits: Tuple[np.ndarray, np.ndarray],
        lookahead_n: int = 5,
        lookahead_ds: float = 0.02
    ):
        """Initialize observation builder.

        Args:
            n_joints: Number of robot joints
            joint_limits: Tuple of (q_min, q_max) arrays
            lookahead_n: Number of lookahead points
            lookahead_ds: Distance between lookahead points (meters)
        """
        self.n_joints = n_joints
        self.q_min, self.q_max = joint_limits
        self.lookahead_n = lookahead_n
        self.lookahead_ds = lookahead_ds

        # Calculate expected observation dimension
        self.obs_dim = (
            3 +  # position error
            3 +  # orientation error
            lookahead_n * 3 +  # lookahead points
            3 +  # reference velocity
            3 +  # EE linear velocity
            3 +  # EE angular velocity
            n_joints +  # joint positions
            n_joints +  # joint velocities
            6 +  # previous action
            1 +  # manipulability
            n_joints  # joint limit proximity
        )

    def build(
        self,
        state: RobotState,
        path: Path,
        s_current: float,
        include_orientation: bool = False
    ) -> np.ndarray:
        """Build observation vector.

        Args:
            state: Current robot state
            path: Path being tracked
            s_current: Current arc length on path
            include_orientation: Whether to include orientation error (Step 4)

        Returns:
            obs: (obs_dim,) observation vector

        Note:
            All path-relative quantities are in EE frame for generalization.
        """
        # EE frame rotation: world -> EE
        R_ew = state.ee_rot_world.T  # Transpose = inverse for rotation

        # (a) Position error in EE frame
        p_target = path.position(s_current)
        pos_err_world = p_target - state.ee_pos_world
        pos_err_ee = R_ew @ pos_err_world  # (3,)

        # (b) Orientation error in EE frame
        if include_orientation:
            q_target = path.orientation(s_current)
            # Rotation error: current -> target
            ori_err_rotvec = rotation_error_rotvec(state.ee_quat_world, q_target)
            # Transform to EE frame
            ori_err_ee = R_ew @ ori_err_rotvec  # (3,)
        else:
            # Step 1: no orientation error (fill with zeros)
            ori_err_ee = np.zeros(3)

        # (c) Lookahead points in EE frame
        lookahead_world = path.lookahead_points(s_current, self.lookahead_n, self.lookahead_ds)
        lookahead_ee = []
        for p_k in lookahead_world:
            # Relative to current EE position, in EE frame
            p_rel_ee = R_ew @ (p_k - state.ee_pos_world)
            lookahead_ee.append(p_rel_ee)
        lookahead_ee = np.concatenate(lookahead_ee)  # (lookahead_n * 3,)

        # (d) Reference velocity in EE frame
        v_ref_world = path.velocity(s_current)
        v_ref_ee = R_ew @ v_ref_world  # (3,)

        # (e) EE velocities in EE frame
        ee_lin_vel_ee = R_ew @ state.ee_lin_vel_world  # (3,)
        ee_ang_vel_ee = R_ew @ state.ee_ang_vel_world  # (3,)

        # (f) Joint state (world frame, but independent of workspace position)
        q_pos = state.joint_pos.copy()  # (n,)
        q_vel = state.joint_vel.copy()  # (n,)

        # (g) Previous action
        prev_action = state.prev_action.copy()  # (6,)

        # (h) Manipulability (scalar)
        manip = compute_manipulability(state.jacobian)  # scalar

        # (i) Joint limit proximity
        limit_prox = joint_limit_proximity(q_pos, (self.q_min, self.q_max))  # (n,)

        # Assemble observation
        obs = np.concatenate([
            pos_err_ee,       # 3
            ori_err_ee,       # 3
            lookahead_ee,     # lookahead_n * 3
            v_ref_ee,         # 3
            ee_lin_vel_ee,    # 3
            ee_ang_vel_ee,    # 3
            q_pos,            # n
            q_vel,            # n
            prev_action,      # 6
            [manip],          # 1
            limit_prox,       # n
        ]).astype(np.float32)

        assert obs.shape == (self.obs_dim,), \
            f"Observation shape mismatch: {obs.shape} vs ({self.obs_dim},)"

        return obs

    def observation_space_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get observation space bounds for Gymnasium.

        Returns:
            Tuple of (low, high) bounds for observation space

        Note:
            Using conservative bounds. Can be tightened later if needed.
        """
        # Most quantities are unbounded in practice, use large values
        low = -np.inf * np.ones(self.obs_dim, dtype=np.float32)
        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)

        # Joint positions have known limits
        joint_pos_start = (
            3 +  # pos error
            3 +  # ori error
            self.lookahead_n * 3 +  # lookahead
            3 +  # ref vel
            3 +  # ee lin vel
            3    # ee ang vel
        )
        joint_pos_end = joint_pos_start + self.n_joints
        low[joint_pos_start:joint_pos_end] = self.q_min
        high[joint_pos_start:joint_pos_end] = self.q_max

        # Manipulability is >= 0
        manip_idx = self.obs_dim - 1 - self.n_joints
        low[manip_idx] = 0.0

        # Joint limit proximity is in [0, 1]
        limit_prox_start = self.obs_dim - self.n_joints
        low[limit_prox_start:] = 0.0
        high[limit_prox_start:] = 1.0

        return low, high
