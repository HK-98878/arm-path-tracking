"""Tracking reward computation with bounded exponential rewards.

Implements progressive reward structure:
- Step 1: Position only
- Step 3: Add velocity matching
- Step 4: Add orientation

All rewards use bounded exponential form to encourage settling.
"""

import numpy as np
from typing import Optional
from ..utils.kinematics import geodesic_angle


class TrackingReward:
    """Compute tracking reward with configurable terms."""

    def __init__(
        self,
        w_pos: float = 1.0,
        w_ori: float = 0.0,
        w_vel: float = 0.0,
        w_action_rate: float = 0.01,
        w_joint_vel: float = 0.001,
        sig_pos: float = 0.02,
        sig_ori: float = 0.1,
        sig_vel: float = 0.05
    ):
        """Initialize reward computer.

        Args:
            w_pos: Weight for position reward
            w_ori: Weight for orientation reward (0 for Step 1)
            w_vel: Weight for velocity matching reward (0 for Step 1-2)
            w_action_rate: Weight for action rate penalty
            w_joint_vel: Weight for joint velocity penalty
            sig_pos: Position error standard deviation (meters)
            sig_ori: Orientation error standard deviation (radians)
            sig_vel: Velocity error standard deviation (m/s)

        Note:
            Step 1 (baseline): w_ori=0, w_vel=0
            Step 3 (velocity): w_vel > 0
            Step 4 (orientation): w_ori > 0
        """
        self.w_pos = w_pos
        self.w_ori = w_ori
        self.w_vel = w_vel
        self.w_action_rate = w_action_rate
        self.w_joint_vel = w_joint_vel

        self.sig_pos = sig_pos
        self.sig_ori = sig_ori
        self.sig_vel = sig_vel

    def compute(
        self,
        pos_error: float,
        action: np.ndarray,
        prev_action: np.ndarray,
        joint_vel: np.ndarray,
        ori_error: Optional[float] = None,
        vel_error: Optional[float] = None
    ) -> dict:
        """Compute reward and its components.

        Args:
            pos_error: L2 position error (meters)
            action: Current action (6,)
            prev_action: Previous action (6,)
            joint_vel: Joint velocities (n,)
            ori_error: Optional geodesic orientation error (radians)
            vel_error: Optional L2 velocity error (m/s)

        Returns:
            Dictionary with:
            - reward: Total reward
            - r_pos: Position reward component
            - r_ori: Orientation reward component
            - r_vel: Velocity matching reward component
            - p_action_rate: Action rate penalty
            - p_joint_vel: Joint velocity penalty
        """
        # Position reward: bounded exponential
        pos_quality = np.exp(-(pos_error / self.sig_pos) ** 2)
        r_pos = self.w_pos * pos_quality

        # Orientation reward (if enabled)
        if self.w_ori > 0 and ori_error is not None:
            r_ori = self.w_ori * np.exp(-(ori_error / self.sig_ori) ** 2)
        else:
            r_ori = 0.0

        # Velocity matching reward (if enabled)
        # Scale by pos_quality: when off-path, focus on position correction;
        # when on-path, velocity matching kicks in to prevent overshoot
        if self.w_vel > 0 and vel_error is not None:
            effective_w_vel = self.w_vel * pos_quality
            r_vel = effective_w_vel * np.exp(-(vel_error / self.sig_vel) ** 2)
        else:
            r_vel = 0.0

        # Action rate penalty (jerk in action space)
        # Adaptive: scale by pos_quality so corrections aren't penalized when off-path
        action_change = action - prev_action
        effective_w_action_rate = self.w_action_rate * pos_quality
        p_action_rate = effective_w_action_rate * np.sum(action_change ** 2)

        # Joint velocity penalty (catches null-space jitter)
        # Also adaptive for consistency
        effective_w_joint_vel = self.w_joint_vel * pos_quality
        p_joint_vel = effective_w_joint_vel * np.sum(joint_vel ** 2)

        # Total reward
        reward = r_pos + r_ori + r_vel - p_action_rate - p_joint_vel

        return {
            'reward': float(reward),
            'r_pos': float(r_pos),
            'r_ori': float(r_ori),
            'r_vel': float(r_vel),
            'p_action_rate': float(p_action_rate),
            'p_joint_vel': float(p_joint_vel),
        }

    def compute_from_state(
        self,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        ee_vel: np.ndarray,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray],
        target_vel: Optional[np.ndarray],
        action: np.ndarray,
        prev_action: np.ndarray,
        joint_vel: np.ndarray
    ) -> dict:
        """Compute reward from full state (convenience wrapper).

        Args:
            ee_pos: Current EE position (3,)
            ee_quat: Current EE quaternion (4,)
            ee_vel: Current EE linear velocity (3,)
            target_pos: Target position (3,)
            target_quat: Optional target quaternion (4,)
            target_vel: Optional target velocity (3,)
            action: Current action (6,)
            prev_action: Previous action (6,)
            joint_vel: Joint velocities (n,)

        Returns:
            Reward dictionary
        """
        # Position error
        pos_error = np.linalg.norm(ee_pos - target_pos)

        # Orientation error (if needed)
        ori_error = None
        if self.w_ori > 0 and target_quat is not None:
            ori_error = geodesic_angle(ee_quat, target_quat)

        # Velocity error (if needed)
        vel_error = None
        if self.w_vel > 0 and target_vel is not None:
            vel_error = np.linalg.norm(ee_vel - target_vel)

        return self.compute(
            pos_error=pos_error,
            action=action,
            prev_action=prev_action,
            joint_vel=joint_vel,
            ori_error=ori_error,
            vel_error=vel_error
        )

    def max_reward(self) -> float:
        """Get maximum possible reward (perfect tracking, zero penalties).

        Returns:
            Maximum reward value
        """
        return self.w_pos + self.w_ori + self.w_vel

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrackingReward(w_pos={self.w_pos}, w_ori={self.w_ori}, "
            f"w_vel={self.w_vel}, w_action_rate={self.w_action_rate}, "
            f"w_joint_vel={self.w_joint_vel})"
        )
