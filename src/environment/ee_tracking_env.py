"""Gymnasium environment for end-effector tracking with residual control."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path as FilePath
from typing import Optional, Tuple, Dict, Any

import mujoco

from .robot_state import RobotState
from .observation import ObservationBuilder
from ..paths.base_path import Path
from ..control.dls_jacobian import DLSController, compute_jacobian_mujoco
from ..rewards.tracking_reward import TrackingReward
from ..utils.kinematics import quat_to_matrix, rotation_error_rotvec


class EETrackingEnv(gym.Env):
    """End-effector tracking environment with residual-feedforward control.

    Action space: 6-DOF Cartesian twist residual (policy outputs corrections)
    Observation space: 58-dim (position error, velocities, lookahead, etc.)

    Key design:
    - Action = 0 yields pure feedforward tangent following (safe default)
    - DLS Jacobian layer handles redundancy resolution
    - All path-relative quantities in EE frame for generalization
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        model_path: str,
        path: Path,
        reward_config: dict,
        action_scale: np.ndarray = np.array([0.02, 0.02, 0.02, 0.10, 0.10, 0.10]),
        dt: float = 0.01,
        max_episode_steps: int = 1000,
        ee_body_name: str = "attachment_site",
        render_mode: Optional[str] = None,
        dls_config: Optional[dict] = None,
        include_orientation: bool = False,
        lookahead_ds: float = 0.02,
        randomize_start_position: bool = False,
        start_position_noise: float = 0.06,
    ):
        """Initialize environment.

        Args:
            model_path: Path to MuJoCo XML model
            path: Path object to track
            reward_config: Dictionary with reward weights
            action_scale: Action scaling [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
            dt: Control timestep (seconds)
            max_episode_steps: Maximum steps per episode
            ee_body_name: Name of end-effector body in MuJoCo model
            render_mode: Rendering mode (None for headless)
            dls_config: Optional DLS controller config
            include_orientation: Whether to include orientation control
            lookahead_ds: Distance between lookahead points in observation (meters)
            randomize_start_position: Sample a random arc-length phase and solve IK each reset
            start_position_noise: Sphere radius (m) for positional offset around the path point
        """
        super().__init__()

        self.render_mode = render_mode
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.action_scale = np.array(action_scale, dtype=np.float32)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Set timestep
        self.model.opt.timestep = dt

        # Get end-effector body ID
        self.ee_body_name = ee_body_name
        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            ee_body_name
        )

        # Number of joints
        self.n_joints = self.model.nv

        # Joint limits
        self.q_min = self.model.jnt_range[:self.n_joints, 0]
        self.q_max = self.model.jnt_range[:self.n_joints, 1]

        # Path
        self.path = path
        self.s_current = 0.0  # Arc length

        # DLS controller
        dls_config = dls_config or {}
        self.dls_controller = DLSController(
            n_joints=self.n_joints,
            q_limits=(self.q_min, self.q_max),
            **dls_config
        )

        # Orientation control flag
        self.include_orientation = include_orientation

        # Observation builder
        self.obs_builder = ObservationBuilder(
            n_joints=self.n_joints,
            joint_limits=(self.q_min, self.q_max),
            lookahead_ds=lookahead_ds
        )

        self.randomize_start_position = randomize_start_position
        self.start_position_noise = start_position_noise

        # Reward computer
        self.reward_computer = TrackingReward(**reward_config)

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        obs_low, obs_high = self.obs_builder.observation_space_bounds()
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # Rendering setup
        self.renderer = None
        self.camera = None
        if render_mode is not None:
            self._setup_rendering(render_mode)

        # Episode tracking
        self.step_count = 0
        self.prev_action = np.zeros(6, dtype=np.float32)

    _Q_NOMINAL = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])

    def _solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        n_iters: int = 100,
        pos_tol: float = 0.002,
    ) -> Optional[np.ndarray]:
        """Iterative 6-DOF IK via DLS, seeded from nominal q. Returns q or None."""
        q = self._Q_NOMINAL.copy()
        for _ in range(n_iters):
            self.data.qpos[:self.n_joints] = q
            self.data.qvel[:self.n_joints] = 0.0
            mujoco.mj_forward(self.model, self.data)
            pos_err = target_pos - self.data.xpos[self.ee_body_id]
            if np.linalg.norm(pos_err) < pos_tol:
                return q
            ee_quat_mj = self.data.xquat[self.ee_body_id]
            ee_quat = np.array([ee_quat_mj[1], ee_quat_mj[2], ee_quat_mj[3], ee_quat_mj[0]])
            ori_err = rotation_error_rotvec(ee_quat, target_quat_xyzw)
            J = compute_jacobian_mujoco(self.model, self.data, self.ee_body_id)
            dq = self.dls_controller.compute_dq(
                dx_ee_world=np.concatenate([pos_err, ori_err]),
                q_current=q,
                jacobian=J,
                use_null_space=True,
            )
            q = np.clip(q + dq, self.q_min, self.q_max)
        self.data.qpos[:self.n_joints] = q
        mujoco.mj_forward(self.model, self.data)
        if np.linalg.norm(target_pos - self.data.xpos[self.ee_body_id]) < pos_tol * 5:
            return q
        return None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed
            options: Optional reset options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Select orientation mode for this episode (if path supports it)
        if hasattr(self.path, 'reset_orientation_mode'):
            self.path.reset_orientation_mode(self.np_random)

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Compute initial joint configuration
        if self.randomize_start_position:
            s_phase = self.np_random.uniform(0.0, self.path.total_length)
            p_on_path = self.path.position(s_phase)
            noise_dir = self.np_random.standard_normal(3)
            noise_dir /= np.linalg.norm(noise_dir)
            noise_mag = self.np_random.uniform(0.0, self.start_position_noise)
            target_pos = p_on_path + noise_dir * noise_mag
            target_quat = self.path.orientation(s_phase)
            result = self._solve_ik(target_pos, target_quat)
            q_init = result if result is not None else self._Q_NOMINAL
        else:
            q_init = self._Q_NOMINAL

        self.data.qpos[:self.n_joints] = q_init
        self.data.qvel[:self.n_joints] = 0.0
        self.data.ctrl[:self.n_joints] = q_init
        mujoco.mj_forward(self.model, self.data)

        # Find closest point on path to start tracking
        ee_pos = self.data.xpos[self.ee_body_id]

        s_samples = np.linspace(0, self.path.total_length, 100)
        errors = [np.linalg.norm(ee_pos - self.path.position(s)) for s in s_samples]
        closest_idx = np.argmin(errors)
        self.s_current = s_samples[closest_idx]

        # Print initial EE position for debugging (first reset only)
        if not hasattr(self, '_printed_init_pos'):
            target_pos = self.path.position(self.s_current)
            init_error = np.linalg.norm(ee_pos - target_pos)
            s0_pos = self.path.position(0)
            print(f"\n  Initial EE position: {ee_pos}")
            print(f"  Circle center: {self.path.center}, radius: {self.path.radius}")
            print(f"  Position at s=0: {s0_pos}")
            print(f"  Starting at s={self.s_current:.3f} / {self.path.total_length:.3f} (closest point)")
            print(f"  Target position: {target_pos}")
            print(f"  Initial error: {init_error*1000:.1f}mm")
            self._printed_init_pos = True

        # Reset episode state
        self.step_count = 0
        self.prev_action = np.zeros(6, dtype=np.float32)

        # Initialize desired joint position for integrated position control
        self.q_desired = self.data.qpos[:self.n_joints].copy()

        # Initialize ideal path position (advances at constant speed, caps s_current)
        self.s_ideal = self.s_current

        # Get initial observation
        state = self._get_robot_state()
        obs = self.obs_builder.build(state, self.path, self.s_current,
                                     include_orientation=self.include_orientation)

        info = {
            's': self.s_current,
            'step': self.step_count,
            'orientation_mode': getattr(self.path, '_orientation_mode', 'fixed')
        }

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Policy action ∈ [-1, 1]^6 (Cartesian twist residual)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to target EE twist (residual-feedforward)
        dx_ee_world = self._action_to_ee_twist(action)

        # DLS layer: EE twist -> joint velocity command
        state = self._get_robot_state()
        dq = self.dls_controller.compute_dq(
            dx_ee_world=dx_ee_world,
            q_current=state.joint_pos,
            jacobian=state.jacobian,
            use_null_space=True
        )

        # Integrate desired position (not based on current pos, to avoid lag)
        self.q_desired = self.q_desired + dq
        self.q_desired = np.clip(self.q_desired, self.q_min, self.q_max)

        # Execute command (position control via MuJoCo actuators)
        self.data.ctrl[:self.n_joints] = self.q_desired

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Update path reference (open-loop advancement)
        # Track arc-length progress for velocity reward
        s_previous = self.s_current
        self._advance_path_reference()

        # Compute arc-length progress (ds/dt normalized by target speed)
        # s_current is unwrapped, so ds is simply the difference (no wrap handling needed)
        ds = self.s_current - s_previous
        s_wrapped = self.s_current % self.path.total_length
        target_speed = np.linalg.norm(self.path.velocity(s_wrapped))
        arc_progress = ds / (self.dt * target_speed) if target_speed > 1e-8 else 0.0

        # Get new state
        state_new = self._get_robot_state()
        state_new.prev_action = action  # Update previous action

        # Compute observation (use wrapped s for path lookups)
        obs = self.obs_builder.build(state_new, self.path, s_wrapped,
                                     include_orientation=self.include_orientation)

        # Compute reward
        target_pos = self.path.position(s_wrapped)
        target_vel = self.path.velocity(s_wrapped)
        target_quat = self.path.orientation(s_wrapped) if self.include_orientation else None

        reward_dict = self.reward_computer.compute_from_state(
            ee_pos=state_new.ee_pos_world,
            ee_quat=state_new.ee_quat_world,
            ee_vel=state_new.ee_lin_vel_world,
            target_pos=target_pos,
            target_quat=target_quat,
            target_vel=target_vel,
            action=action,
            prev_action=self.prev_action,
            joint_vel=state_new.joint_vel,
            arc_progress=arc_progress
        )

        reward = reward_dict['reward']

        # Update state
        self.prev_action = action.copy()
        self.step_count += 1

        # Episode termination
        terminated = False  # No early termination for now
        truncated = self.step_count >= self.max_episode_steps

        # Info
        pos_error = np.linalg.norm(state_new.ee_pos_world - target_pos)
        info = {
            's': s_wrapped,  # Report wrapped arc length
            'step': self.step_count,
            'pos_error': pos_error,
            **reward_dict,  # Include reward components
        }

        return obs, reward, terminated, truncated, info

    def _get_robot_state(self) -> RobotState:
        """Extract robot state from MuJoCo data.

        Returns:
            RobotState object
        """
        # Joint state
        joint_pos = self.data.qpos[:self.n_joints].copy()
        joint_vel = self.data.qvel[:self.n_joints].copy()

        # End-effector pose
        ee_pos_world = self.data.xpos[self.ee_body_id].copy()
        ee_quat_world = self.data.xquat[self.ee_body_id].copy()  # [w,x,y,z] in MuJoCo
        # Convert to [x,y,z,w] for scipy
        ee_quat_world = np.array([
            ee_quat_world[1],
            ee_quat_world[2],
            ee_quat_world[3],
            ee_quat_world[0]
        ])
        ee_rot_world = quat_to_matrix(ee_quat_world)

        # End-effector velocity (from body)
        ee_lin_vel_world = self.data.cvel[self.ee_body_id, 3:6].copy()  # Linear
        ee_ang_vel_world = self.data.cvel[self.ee_body_id, 0:3].copy()  # Angular

        # Jacobian
        jacobian = compute_jacobian_mujoco(self.model, self.data, self.ee_body_id)

        return RobotState(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            ee_pos_world=ee_pos_world,
            ee_quat_world=ee_quat_world,
            ee_rot_world=ee_rot_world,
            ee_lin_vel_world=ee_lin_vel_world,
            ee_ang_vel_world=ee_ang_vel_world,
            jacobian=jacobian,
            prev_action=self.prev_action.copy()
        )

    def _action_to_ee_twist(self, action: np.ndarray) -> np.ndarray:
        """Convert policy action to EE twist in world frame.

        Args:
            action: Policy action ∈ [-1, 1]^6 (in EE frame)

        Returns:
            dx_ee_world: (6,) EE twist * dt in world frame

        Note:
            Residual-feedforward formulation:
            dx_ee = feedforward_step + R_we @ (action_scale * action)

            The policy observes errors in EE frame, so its corrections are
            in EE frame. We transform them to world frame before adding to
            the feedforward (which is already in world frame).
        """
        # Get current EE rotation for frame transformation
        ee_quat_mj = self.data.xquat[self.ee_body_id]  # [w,x,y,z] MuJoCo format
        ee_quat = np.array([ee_quat_mj[1], ee_quat_mj[2], ee_quat_mj[3], ee_quat_mj[0]])
        R_we = quat_to_matrix(ee_quat)  # world <- ee rotation

        # Feedforward: tangent step from path geometry (world frame)
        s_wrapped = self.s_current % self.path.total_length
        v_ref = self.path.velocity(s_wrapped)
        feedforward_linear = v_ref * self.dt

        # Angular feedforward: path angular velocity if orientation control enabled
        if self.include_orientation:
            omega_ref = self.path.angular_velocity(s_wrapped)
            feedforward_angular = omega_ref * self.dt
        else:
            feedforward_angular = np.zeros(3)

        # Residual: policy output is in EE frame, transform to world frame
        residual_ee = self.action_scale * action
        residual_linear_world = R_we @ residual_ee[:3]
        residual_angular_world = R_we @ residual_ee[3:]

        # Total twist in world frame
        dx_ee_world = np.concatenate([
            feedforward_linear + residual_linear_world,
            feedforward_angular + residual_angular_world
        ])

        return dx_ee_world

    def _advance_path_reference(self):
        """Advance path reference with bounded nearest-point search.

        Key properties:
        - s_ideal advances at constant path speed (the "clock")
        - s_current tracks nearest point but can't exceed s_ideal (no lapping)
        - When behind: target waits for EE (forgiving for learning)
        - When caught up: target moves at constant speed (proper tracking)
        - Supports both forward (positive speed) and reverse (negative speed) paths

        Note: s_ideal and s_current are tracked as unwrapped values (can exceed
        total_length for multiple laps). Only wrap when accessing path geometry.
        """
        ee_pos = self.data.xpos[self.ee_body_id]

        # Check if path is reversed (negative speed)
        is_reversed = self.path.speed < 0

        # Advance ideal position at constant path speed (unwrapped)
        # Use signed speed for direction, magnitude for rate
        s_ideal_wrapped = self.s_ideal % self.path.total_length
        speed_magnitude = np.linalg.norm(self.path.velocity(s_ideal_wrapped))
        if is_reversed:
            self.s_ideal = self.s_ideal - speed_magnitude * self.dt
        else:
            self.s_ideal = self.s_ideal + speed_magnitude * self.dt

        # Nearest-point search (allows catching up when behind)
        backward_allowance = 0.0
        max_step = 0.05  # Max 50mm per step
        n_samples = 100

        # Search direction depends on path direction
        if is_reversed:
            # Reversed: search backward from current
            s_min = self.s_current - max_step
            s_max = self.s_current + backward_allowance
        else:
            # Forward: search forward from current
            s_min = self.s_current - backward_allowance
            s_max = self.s_current + max_step

        s_samples = np.linspace(s_min, s_max, n_samples)

        # Find nearest point, preferring direction-appropriate s for stability
        min_dist = float('inf')
        best_s = self.s_current
        for s_raw in s_samples:
            # Wrap for position lookup only
            s_wrapped = s_raw % self.path.total_length
            pos = self.path.position(s_wrapped)
            dist = np.linalg.norm(ee_pos - pos)

            if dist < min_dist - 1e-6:
                min_dist = dist
                best_s = s_raw
            elif abs(dist - min_dist) < 1e-6:
                # Prefer smaller s for forward, larger s for reverse
                if (not is_reversed and s_raw < best_s) or (is_reversed and s_raw > best_s):
                    best_s = s_raw

        # Gradient refinement
        best_s = self._refine_nearest_s(ee_pos, best_s)

        # Apply lookahead (in direction of travel)
        lookahead = 0.005
        if is_reversed:
            s_candidate = best_s - lookahead
            # KEY: Clip to s_ideal - can't go beyond the "clock" (reversed)
            self.s_current = max(s_candidate, self.s_ideal)
        else:
            s_candidate = best_s + lookahead
            # KEY: Clip to s_ideal - can't advance beyond the "clock"
            self.s_current = min(s_candidate, self.s_ideal)

    def _refine_nearest_s(self, ee_pos: np.ndarray, s_coarse: float, n_iters: int = 3) -> float:
        """Gradient-based refinement of nearest arc length.

        Uses Newton-like steps to find local minimum of distance to path.
        """
        s = s_coarse
        step_size = 0.001  # 1mm initial step for finite differences

        for _ in range(n_iters):
            s_wrapped = s % self.path.total_length

            # Compute gradient via finite differences
            pos_center = self.path.position(s_wrapped)
            pos_plus = self.path.position((s + step_size) % self.path.total_length)
            pos_minus = self.path.position((s - step_size) % self.path.total_length)

            dist_center = np.linalg.norm(ee_pos - pos_center)
            dist_plus = np.linalg.norm(ee_pos - pos_plus)
            dist_minus = np.linalg.norm(ee_pos - pos_minus)

            # Gradient: d(dist)/ds ≈ (dist_plus - dist_minus) / (2 * step_size)
            grad = (dist_plus - dist_minus) / (2 * step_size)

            # Simple gradient descent step (with damping)
            s = s - 0.5 * step_size * np.sign(grad)

        return s

    def _setup_rendering(self, render_mode: str):
        """Initialize rendering context.

        Args:
            render_mode: "rgb_array" for offscreen rendering, "human" for viewer
        """
        if render_mode == "rgb_array":
            # Offscreen renderer for video export
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.camera = mujoco.MjvCamera()
            self._configure_camera()
        elif render_mode == "human":
            # Interactive viewer mode - handled by visualize_policy.py
            # Just set up camera for potential use
            self.camera = mujoco.MjvCamera()
            self._configure_camera()

    def _configure_camera(self):
        """Configure default camera view centered on path."""
        # Set camera to view the workspace around the path
        # For circle path, center on path center
        if hasattr(self.path, 'center'):
            self.camera.lookat = self.path.center
            # Set distance based on path size
            if hasattr(self.path, 'radius'):
                self.camera.distance = 2.5 * self.path.radius + 0.5
            else:
                self.camera.distance = 1.5
        else:
            # Default view for other paths
            self.camera.lookat = np.array([0.4, 0.0, 0.5])
            self.camera.distance = 1.5

        self.camera.azimuth = 45
        self.camera.elevation = -20

    def render(self):
        """Render environment.

        Returns:
            rgb_array: (H, W, 3) numpy array if render_mode is "rgb_array"
            None: for "human" mode (handled externally)
        """
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                raise RuntimeError("Renderer not initialized. Set render_mode in constructor.")
            self.renderer.update_scene(self.data, camera=self.camera)
            return self.renderer.render()
        return None

    def close(self):
        """Clean up resources."""
        pass
