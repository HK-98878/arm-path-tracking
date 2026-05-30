"""Robot state dataclass for managing kinematic and dynamic state."""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class RobotState:
    """Container for robot kinematic and dynamic state.

    This class holds all state information needed by the control layer
    and observation builder.
    """

    # Joint state
    joint_pos: np.ndarray  # (n,) joint positions (rad)
    joint_vel: np.ndarray  # (n,) joint velocities (rad/s)

    # End-effector state (world frame)
    ee_pos_world: np.ndarray  # (3,) EE position
    ee_quat_world: np.ndarray  # (4,) EE orientation quaternion [x,y,z,w]
    ee_rot_world: np.ndarray  # (3,3) EE rotation matrix (world <- EE)

    ee_lin_vel_world: np.ndarray  # (3,) EE linear velocity
    ee_ang_vel_world: np.ndarray  # (3,) EE angular velocity

    # Jacobian (world frame)
    jacobian: np.ndarray  # (6, n) geometric Jacobian [J_v; J_w]

    # Previous action (for observation)
    prev_action: np.ndarray  # (6,) previous action (required for CAPS)

    # Optional diagnostics
    manipulability: Optional[float] = None
    near_singularity: Optional[bool] = None

    def __post_init__(self):
        """Validate dimensions after initialization."""
        n_joints = len(self.joint_pos)

        assert self.joint_vel.shape == (n_joints,), \
            f"joint_vel shape mismatch: {self.joint_vel.shape} vs ({n_joints},)"

        assert self.ee_pos_world.shape == (3,), \
            f"ee_pos_world shape mismatch: {self.ee_pos_world.shape}"

        assert self.ee_quat_world.shape == (4,), \
            f"ee_quat_world shape mismatch: {self.ee_quat_world.shape}"

        assert self.ee_rot_world.shape == (3, 3), \
            f"ee_rot_world shape mismatch: {self.ee_rot_world.shape}"

        assert self.ee_lin_vel_world.shape == (3,), \
            f"ee_lin_vel_world shape mismatch: {self.ee_lin_vel_world.shape}"

        assert self.ee_ang_vel_world.shape == (3,), \
            f"ee_ang_vel_world shape mismatch: {self.ee_ang_vel_world.shape}"

        assert self.jacobian.shape == (6, n_joints), \
            f"jacobian shape mismatch: {self.jacobian.shape} vs (6, {n_joints})"

        assert self.prev_action.shape == (6,), \
            f"prev_action shape mismatch: {self.prev_action.shape}"

    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return len(self.joint_pos)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/debugging."""
        return {
            'joint_pos': self.joint_pos.copy(),
            'joint_vel': self.joint_vel.copy(),
            'ee_pos_world': self.ee_pos_world.copy(),
            'ee_quat_world': self.ee_quat_world.copy(),
            'ee_lin_vel_world': self.ee_lin_vel_world.copy(),
            'ee_ang_vel_world': self.ee_ang_vel_world.copy(),
            'manipulability': self.manipulability,
            'near_singularity': self.near_singularity,
        }
