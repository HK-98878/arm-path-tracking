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
from ..utils.kinematics import quat_to_matrix


class EETrackingEnv(gym.Env):
    """End-effector tracking environment with residual-feedforward control.

    Action space: 6-DOF Cartesian twist residual (policy outputs corrections)
    Observation space: 58-dim (position error, velocities, lookahead, etc.)

    Key design:
    - Action = 0 yields pure feedforward tangent following (safe default)
    - DLS Jacobian layer handles redundancy resolution
    - All path-relative quantities in EE frame for generalization
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

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
        dls_config: Optional[dict] = None
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

        # Observation builder
        self.obs_builder = ObservationBuilder(
            n_joints=self.n_joints,
            joint_limits=(self.q_min, self.q_max)
        )

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

        # Episode tracking
        self.step_count = 0
        self.prev_action = np.zeros(6, dtype=np.float32)

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

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set initial joint configuration
        # Use a config that puts EE closer to the circle
        # This is a manually tuned configuration for circle at [0.5, 0, 0.3] radius 0.3
        q_init = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])
        self.data.qpos[:self.n_joints] = q_init
        self.data.qvel[:self.n_joints] = 0.0

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Find closest point on path to start tracking
        ee_pos = self.data.xpos[self.ee_body_id]

        # Search for closest arc length
        s_samples = np.linspace(0, self.path.total_length, 100)
        errors = []
        for s in s_samples:
            pos = self.path.position(s)
            error = np.linalg.norm(ee_pos - pos)
            errors.append(error)

        closest_idx = np.argmin(errors)
        self.s_current = s_samples[closest_idx]

        # Print initial EE position for debugging (first reset only)
        if not hasattr(self, '_printed_init_pos'):
            target_pos = self.path.position(self.s_current)
            init_error = np.linalg.norm(ee_pos - target_pos)
            print(f"\n  Initial EE position: {ee_pos}")
            print(f"  Starting at s={self.s_current:.3f} (closest point)")
            print(f"  Target position: {target_pos}")
            print(f"  Initial error: {init_error*1000:.1f}mm")
            self._printed_init_pos = True

        # Reset episode state
        self.step_count = 0
        self.prev_action = np.zeros(6, dtype=np.float32)

        # Get initial observation
        state = self._get_robot_state()
        obs = self.obs_builder.build(state, self.path, self.s_current)

        info = {
            's': self.s_current,
            'step': self.step_count
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

        # DLS layer: EE twist -> joint command
        state = self._get_robot_state()
        q_cmd = self.dls_controller.step(
            dx_ee_world=dx_ee_world,
            q_current=state.joint_pos,
            jacobian=state.jacobian,
            use_null_space=True
        )

        # Execute command (position control via MuJoCo actuators)
        self.data.ctrl[:self.n_joints] = q_cmd

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Update path reference (open-loop advancement)
        self._advance_path_reference()

        # Get new state
        state_new = self._get_robot_state()
        state_new.prev_action = action  # Update previous action

        # Compute observation
        obs = self.obs_builder.build(state_new, self.path, self.s_current)

        # Compute reward
        target_pos = self.path.position(self.s_current)
        target_vel = self.path.velocity(self.s_current)

        reward_dict = self.reward_computer.compute_from_state(
            ee_pos=state_new.ee_pos_world,
            ee_quat=state_new.ee_quat_world,
            ee_vel=state_new.ee_lin_vel_world,
            target_pos=target_pos,
            target_quat=None,  # Step 1: no orientation
            target_vel=target_vel,  # Enable velocity matching
            action=action,
            prev_action=self.prev_action,
            joint_vel=state_new.joint_vel
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
            's': self.s_current,
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
        v_ref = self.path.velocity(self.s_current)
        feedforward_linear = v_ref * self.dt

        # For Step 1: no angular feedforward (no orientation tracking)
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
        """Advance path reference using nearest-point (carrot-on-stick).

        Instead of open-loop advancement which can outrun the robot,
        we find the nearest point on the path and add a small lookahead.
        This ensures the reference "waits" for the robot if it falls behind.
        """
        # Get current EE position
        ee_pos = self.data.xpos[self.ee_body_id]

        # Find nearest point on path (search around current s)
        # Search in a window around current position for efficiency
        search_range = 0.2  # Search ±0.2m of arc length
        n_samples = 50
        s_min = self.s_current - search_range
        s_max = self.s_current + search_range

        s_samples = np.linspace(s_min, s_max, n_samples)
        # Wrap to valid range [0, total_length)
        s_samples = s_samples % self.path.total_length

        # Find closest point
        min_dist = float('inf')
        nearest_s = self.s_current
        for s in s_samples:
            pos = self.path.position(s)
            dist = np.linalg.norm(ee_pos - pos)
            if dist < min_dist:
                min_dist = dist
                nearest_s = s

        # Add small lookahead so reference stays slightly ahead
        lookahead = 0.01  # 10mm lookahead in arc length
        self.s_current = (nearest_s + lookahead) % self.path.total_length

    def render(self):
        """Render environment (optional, for debugging)."""
        if self.render_mode == "rgb_array":
            # TODO: Implement camera rendering
            pass
        return None

    def close(self):
        """Clean up resources."""
        pass
