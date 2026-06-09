"""Observation builder for end-effector tracking environment.

Constructs the observation vector (for n=7 joints):
- Position error (EE frame): 3
- Orientation error (rotation vector, EE frame): 3
- Lookahead points (EE frame): 15 (5 points × 3)
- Reference velocity (EE frame): 3
- EE linear velocity (EE frame): 3
- EE angular velocity (EE frame): 3
- EE linear acceleration (EE frame): 3  [only if include_ee_accel=True]
- Joint positions: 7
- Joint velocities: 7
- Previous action: 6
- Manipulability: 1
- Joint limit proximity: 7

Total: 58 dimensions (without EE accel) or 61 (with EE accel)
"""

import numpy as np
from typing import Dict, Optional, Tuple
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
        lookahead_ds: float = 0.02,
        curvature_n: int = 0,
        include_ee_accel: bool = False,
        obs_noise_config: Optional[Dict] = None,
        dt: float = 0.01,
    ):
        """Initialize observation builder.

        Args:
            n_joints: Number of robot joints
            joint_limits: Tuple of (q_min, q_max) arrays
            lookahead_n: Number of lookahead points
            lookahead_ds: Distance between lookahead points (meters)
            curvature_n: Number of curvature vectors appended after lookahead points.
                Each is d²r/ds² at arc-length s + (lookahead_n+k)*ds, expressed in
                EE frame (3 dims each). Zero on straights, nonzero at corners.
            include_ee_accel: If True, add 3-dim EE linear acceleration slot
            obs_noise_config: Optional dict with noise std keys:
                obs_noise_pos_std, obs_noise_vel_std, lookahead_noise_std
            dt: Control timestep (seconds), used to compute EE acceleration
        """
        self.n_joints = n_joints
        self.q_min, self.q_max = joint_limits
        self.lookahead_n = lookahead_n
        self.lookahead_ds = lookahead_ds
        self.curvature_n = curvature_n
        self.include_ee_accel = include_ee_accel
        self.obs_noise_config = obs_noise_config or {}
        self.dt = dt

        # Calculate expected observation dimension
        self.obs_dim = (
            3 +  # position error
            3 +  # orientation error
            lookahead_n * 3 +  # lookahead points
            curvature_n * 3 +  # curvature vectors (d²r/ds² at distant arc-length samples)
            3 +  # reference velocity
            3 +  # EE linear velocity
            3 +  # EE angular velocity
            (3 if include_ee_accel else 0) +  # EE linear acceleration (optional)
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
        include_orientation: bool = False,
        prev_ee_vel: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Build observation vector.

        Args:
            state: Current robot state
            path: Path being tracked
            s_current: Current arc length on path
            include_orientation: Whether to include orientation error
            prev_ee_vel: Previous step EE linear velocity (world frame) for
                         acceleration computation; required when include_ee_accel=True
            rng: Random generator for observation noise injection; noise is
                 skipped when None

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
        # ds must be signed: for reversed paths (negative speed), lookahead goes toward lower s
        ds_signed = self.lookahead_ds * np.sign(getattr(path, 'speed', 1.0))
        lookahead_world = path.lookahead_points(s_current, self.lookahead_n, ds_signed)
        lookahead_ee = []
        for p_k in lookahead_world:
            # Relative to current EE position, in EE frame
            p_rel_ee = R_ew @ (p_k - state.ee_pos_world)
            lookahead_ee.append(p_rel_ee)
        lookahead_ee = np.concatenate(lookahead_ee)  # (lookahead_n * 3,)

        # (c2) Curvature vectors in path-tangent frame: d²r/ds² at distant arc-length samples.
        # Approximated as finite tangent difference: (tangent(s_k+ds) - tangent(s_k)) / ds.
        # Sampled at positions immediately beyond the lookahead window.
        # Zero on straight segments; points toward centre of curvature at corners.
        #
        # Expressed in the path-tangent frame at s_current (not EE frame) so the signal
        # is stable as the arm moves: for a circle, the centripetal direction is always
        # [0, 1/R, 0] in this frame regardless of position on the circle.
        curvature_ee = np.empty(0, dtype=np.float64)
        if self.curvature_n > 0:
            # Build path-tangent frame at s_current.
            t_cur = path.tangent(s_current)          # unit tangent
            world_up = np.array([0., 0., 1.])
            if abs(np.dot(t_cur, world_up)) > 0.9:   # near-vertical tangent: fall back
                world_up = np.array([1., 0., 0.])
            n_cur = np.cross(world_up, t_cur)
            n_cur /= np.linalg.norm(n_cur)
            b_cur = np.cross(t_cur, n_cur)
            # R_world_to_path.T = [t_cur | n_cur | b_cur], so R_world_to_path projects world → frame
            R_world_to_path = np.column_stack([t_cur, n_cur, b_cur]).T

            closed = getattr(path, 'closed', True)
            total = path.total_length
            curv_parts = []
            for k in range(self.lookahead_n + 1, self.lookahead_n + 1 + self.curvature_n):
                s_k  = s_current + k * ds_signed
                s_k1 = s_current + (k + 1) * ds_signed
                s_k  = (s_k  % total) if closed else float(np.clip(s_k,  0.0, total))
                s_k1 = (s_k1 % total) if closed else float(np.clip(s_k1, 0.0, total))
                t0 = path.tangent(s_k)
                t1 = path.tangent(s_k1)
                curv_world = (t1 - t0) / abs(self.lookahead_ds)
                curv_parts.append(R_world_to_path @ curv_world)
            curvature_ee = np.concatenate(curv_parts)  # (curvature_n * 3,)

        # (d) Reference velocity in EE frame
        v_ref_world = path.velocity(s_current)
        v_ref_ee = R_ew @ v_ref_world  # (3,)

        # (e) EE velocities in EE frame
        ee_lin_vel_ee = R_ew @ state.ee_lin_vel_world  # (3,)
        ee_ang_vel_ee = R_ew @ state.ee_ang_vel_world  # (3,)

        # (e2) EE linear acceleration in EE frame (optional)
        if self.include_ee_accel:
            if prev_ee_vel is not None:
                ee_accel_world = (state.ee_lin_vel_world - prev_ee_vel) / self.dt
            else:
                ee_accel_world = np.zeros(3)
            ee_accel_ee = R_ew @ ee_accel_world
        else:
            ee_accel_ee = None

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
        parts = [
            pos_err_ee,       # 3
            ori_err_ee,       # 3
            lookahead_ee,     # lookahead_n * 3
            curvature_ee,     # curvature_n * 3  (empty array when curvature_n=0)
            v_ref_ee,         # 3
            ee_lin_vel_ee,    # 3
            ee_ang_vel_ee,    # 3
        ]
        if ee_accel_ee is not None:
            parts.append(ee_accel_ee)   # 3 (optional)
        parts += [
            q_pos,            # n
            q_vel,            # n
            prev_action,      # 6
            [manip],          # 1
            limit_prox,       # n
        ]
        obs = np.concatenate(parts).astype(np.float32)

        assert obs.shape == (self.obs_dim,), \
            f"Observation shape mismatch: {obs.shape} vs ({self.obs_dim},)"

        # Apply observation noise (training robustness)
        if rng is not None and self.obs_noise_config:
            pos_std = self.obs_noise_config.get('obs_noise_pos_std', 0.0)
            if pos_std > 0:
                obs[0:3] += rng.normal(0.0, pos_std, 3).astype(np.float32)

            la_std = self.obs_noise_config.get('lookahead_noise_std', 0.0)
            if la_std > 0:
                obs[6:6 + self.lookahead_n * 3] += rng.normal(
                    0.0, la_std, self.lookahead_n * 3
                ).astype(np.float32)

            vel_std = self.obs_noise_config.get('obs_noise_vel_std', 0.0)
            if vel_std > 0:
                # EE linear velocity starts after: pos(3)+ori(3)+lookahead(n*3)+curvature(n*3)+ref_vel(3)
                vel_start = 3 + 3 + self.lookahead_n * 3 + self.curvature_n * 3 + 3
                obs[vel_start:vel_start + 3] += rng.normal(
                    0.0, vel_std, 3
                ).astype(np.float32)

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
            self.curvature_n * 3 +  # curvature vectors
            3 +  # ref vel
            3 +  # ee lin vel
            3 +  # ee ang vel
            (3 if self.include_ee_accel else 0)  # ee accel (optional)
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
