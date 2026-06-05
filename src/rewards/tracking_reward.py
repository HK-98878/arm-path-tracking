"""Tracking reward computation with bounded exponential rewards.

Implements progressive reward structure:
- Step 1: Position only
- Step 3: Add tangent progress (velocity component along path direction)
- Step 4: Add orientation

Position uses exponential with linear fallback for large errors (ensures gradient
always exists for recovery). Tangent progress is gated by position error to prevent
the policy from just moving along tangent while ignoring position correction.
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
        sig_vel: float = 0.05,
        pos_linear_fallback_max: float = 0.5,
        ori_linear_fallback_max: float = 3.14,
        vel_gate_max: float = 0.3,
        vel_reward_type: str = "tangent",
        w_vel_match: float = 0.0,
        vel_match_gate_max: float = 0.05,
        w_ee_jerk: float = 0.0,
        ee_jerk_scale: float = 1.0,
    ):
        """Initialize reward computer.

        Args:
            w_pos: Weight for position reward
            w_ori: Weight for orientation reward (0 for Step 1)
            w_vel: Weight for velocity/progress reward (0 for Step 1-2)
            w_action_rate: Weight for action rate penalty
            w_joint_vel: Weight for joint velocity penalty
            sig_pos: Position error standard deviation (meters)
            sig_ori: Orientation error standard deviation (radians)
            sig_vel: Velocity scale for tangent progress normalization (m/s)
            pos_linear_fallback_max: Max error for linear fallback (meters). Beyond this,
                                     only minimal position reward. Default 0.5m (500mm).
            ori_linear_fallback_max: Max error for orientation linear fallback (radians).
                                     Provides gradient at large orientation errors where
                                     exponential is near-zero. Default π rad (180°).
            vel_gate_max: Max error for velocity reward gating (meters). Beyond this,
                          velocity reward is zeroed to focus on position correction.
                          Default 0.3m (300mm).
            vel_reward_type: Type of velocity reward:
                - "tangent": Reward velocity component along path tangent (original)
                - "arc_length": Reward arc-length progress ds/dt (new)
                Arc-length progress rewards keeping up with the moving target,
                doesn't reward shortcuts since s only advances when EE is near path.

        Note:
            Step 1 (baseline): w_ori=0, w_vel=0
            Step 3 (velocity progress): w_vel > 0
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
        self.pos_linear_fallback_max = pos_linear_fallback_max
        self.ori_linear_fallback_max = ori_linear_fallback_max
        self.vel_gate_max = vel_gate_max
        self.vel_reward_type = vel_reward_type
        self.w_vel_match = w_vel_match
        self.vel_match_gate_max = vel_match_gate_max
        self.w_ee_jerk = w_ee_jerk
        self.ee_jerk_scale = max(ee_jerk_scale, 1e-8)

    def compute(
        self,
        pos_error: float,
        action: np.ndarray,
        prev_action: np.ndarray,
        joint_vel: np.ndarray,
        ori_error: Optional[float] = None,
        tangent_progress: Optional[float] = None,
        arc_progress: Optional[float] = None,
        vel_error_sq: Optional[float] = None,
        ee_accel_sq: Optional[float] = None,
    ) -> dict:
        """Compute reward and its components.

        Args:
            pos_error: L2 position error (meters)
            action: Current action (6,)
            prev_action: Previous action (6,)
            joint_vel: Joint velocities (n,)
            ori_error: Optional geodesic orientation error (radians)
            tangent_progress: Optional normalized velocity along tangent (ee_vel · tangent_dir / target_speed)
                              Positive = moving with path, negative = moving against
            arc_progress: Optional normalized arc-length progress (ds/dt / target_speed)
                          1.0 = keeping up with path, <1.0 = falling behind, >1.0 = catching up

        Returns:
            Dictionary with:
            - reward: Total reward
            - r_pos: Position reward component
            - r_ori: Orientation reward component
            - r_vel: Velocity/progress reward component
            - p_action_rate: Action rate penalty
            - p_joint_vel: Joint velocity penalty
        """
        # Position reward: exponential with linear fallback for large errors
        # Exponential provides sharp gradient near path, linear ensures gradient exists far from path
        pos_quality_exp = np.exp(-(pos_error / self.sig_pos) ** 2)

        # Linear fallback: decays from 1 at 0 to 0 at fallback_max
        # Provides gradient when exponential is near-zero (at large errors)
        pos_quality_linear = float(np.clip(1.0 - pos_error / self.pos_linear_fallback_max, 0.0, 1.0))

        # Combine: use exponential when close, blend in linear when far
        # The (1 - pos_quality_exp) factor makes linear contribute more as exponential dies off
        pos_quality = pos_quality_exp + 0.3 * pos_quality_linear * (1.0 - pos_quality_exp)
        r_pos = self.w_pos * pos_quality

        # Orientation reward (if enabled)
        # Uses exponential + linear fallback like position, to ensure gradient at large errors
        if self.w_ori > 0 and ori_error is not None:
            ori_quality_exp = np.exp(-(ori_error / self.sig_ori) ** 2)
            ori_quality_linear = float(np.clip(1.0 - ori_error / self.ori_linear_fallback_max, 0.0, 1.0))
            ori_quality = ori_quality_exp + 0.3 * ori_quality_linear * (1.0 - ori_quality_exp)
            r_ori = self.w_ori * ori_quality
        else:
            r_ori = 0.0

        # Velocity/progress reward (if enabled)
        # GATED by position error: when far off-path, focus on position correction first
        r_vel = 0.0
        if self.w_vel > 0:
            # Gate: fades from 1.0 at 0 error to 0.0 at vel_gate_max
            vel_gate = float(np.clip(1.0 - pos_error / self.vel_gate_max, 0.0, 1.0))

            if self.vel_reward_type == "arc_length" and arc_progress is not None:
                # Arc-length progress: rewards keeping up with path advancement
                # arc_progress=1.0 means keeping up, <1.0 falling behind, >1.0 catching up
                # Using tanh to bound the reward and allow catch-up
                # Note: arc_progress naturally stays low when off-path (s doesn't advance)
                # so gating is less critical, but we keep it for consistency
                r_vel = self.w_vel * vel_gate * np.tanh(arc_progress)
            elif self.vel_reward_type == "tangent" and tangent_progress is not None:
                # Tangent progress: rewards velocity component along path tangent
                # tanh maps (-inf, inf) -> (-1, 1)
                # progress=1.0 (at path speed) -> tanh(1) ≈ 0.76
                # progress=2.0 (2x path speed) -> tanh(2) ≈ 0.96
                # progress=0 (stationary) -> 0
                # progress=-1 (backwards at path speed) -> tanh(-1) ≈ -0.76
                r_vel = self.w_vel * vel_gate * np.tanh(tangent_progress)

        # Velocity-matching penalty: penalise squared error vs path reference velocity,
        # gated so it's active near the path but suppressed during corrections
        r_vel_match = 0.0
        if self.w_vel_match > 0 and vel_error_sq is not None:
            vm_gate = float(np.clip(1.0 - pos_error / self.vel_match_gate_max, 0.0, 1.0))
            r_vel_match = -self.w_vel_match * vm_gate * vel_error_sq

        # Action rate penalty (jerk in action space)
        action_change = action - prev_action
        effective_w_action_rate = self.w_action_rate * pos_quality
        p_action_rate = effective_w_action_rate * np.sum(action_change ** 2)

        # Joint velocity penalty (catches null-space jitter)
        # Also adaptive for consistency
        effective_w_joint_vel = self.w_joint_vel * pos_quality
        p_joint_vel = effective_w_joint_vel * np.sum(joint_vel ** 2)

        # EE jerk penalty: penalise EE acceleration squared (physical jerk in workspace).
        # Gated by pos_quality: suppressed during recovery so the policy can accelerate
        # freely when far off-path. Complements CAPS temporal smoothness (which acts on
        # policy outputs) with a direct physical signal.
        p_ee_jerk = 0.0
        if self.w_ee_jerk > 0 and ee_accel_sq is not None:
            p_ee_jerk = self.w_ee_jerk * pos_quality * ee_accel_sq / self.ee_jerk_scale

        # Total reward
        reward = r_pos + r_ori + r_vel + r_vel_match - p_action_rate - p_joint_vel - p_ee_jerk

        return {
            'reward': float(reward),
            'r_pos': float(r_pos),
            'r_ori': float(r_ori),
            'r_vel': float(r_vel),
            'r_vel_match': float(r_vel_match),
            'p_action_rate': float(p_action_rate),
            'p_joint_vel': float(p_joint_vel),
            'p_ee_jerk': float(p_ee_jerk),
            'pos_error': float(pos_error),
            'ori_error': float(ori_error) if ori_error is not None else 0.0,
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
        joint_vel: np.ndarray,
        arc_progress: Optional[float] = None,
        prev_ee_vel: Optional[np.ndarray] = None,
        dt: float = 0.01,
    ) -> dict:
        """Compute reward from full state (convenience wrapper).

        Args:
            ee_pos: Current EE position (3,)
            ee_quat: Current EE quaternion (4,)
            ee_vel: Current EE linear velocity (3,)
            target_pos: Target position (3,)
            target_quat: Optional target quaternion (4,)
            target_vel: Optional target velocity (3,) - path tangent velocity
            action: Current action (6,)
            prev_action: Previous action (6,)
            joint_vel: Joint velocities (n,)
            arc_progress: Optional arc-length progress (ds/dt / target_speed)

        Returns:
            Reward dictionary
        """
        # Position error
        pos_error = np.linalg.norm(ee_pos - target_pos)

        # Orientation error (if needed)
        ori_error = None
        if self.w_ori > 0 and target_quat is not None:
            ori_error = geodesic_angle(ee_quat, target_quat)

        # Tangent progress (if needed for tangent reward type)
        # Compute velocity component along path tangent, normalized by target speed
        tangent_progress = None
        if self.w_vel > 0 and self.vel_reward_type == "tangent" and target_vel is not None:
            target_speed = np.linalg.norm(target_vel)
            if target_speed > 1e-8:
                tangent_dir = target_vel / target_speed
                # Dot product gives velocity component along tangent
                progress = np.dot(ee_vel, tangent_dir)
                # Normalize by target speed so progress=1.0 means "keeping up"
                tangent_progress = progress / target_speed
            else:
                tangent_progress = 0.0

        vel_error_sq = None
        if self.w_vel_match > 0 and target_vel is not None:
            vel_error_sq = float(np.sum((ee_vel - target_vel) ** 2))

        ee_accel_sq = None
        if self.w_ee_jerk > 0 and prev_ee_vel is not None:
            ee_accel = (ee_vel - prev_ee_vel) / dt
            ee_accel_sq = float(np.sum(ee_accel ** 2))

        return self.compute(
            pos_error=pos_error,
            action=action,
            prev_action=prev_action,
            joint_vel=joint_vel,
            ori_error=ori_error,
            tangent_progress=tangent_progress,
            arc_progress=arc_progress,
            vel_error_sq=vel_error_sq,
            ee_accel_sq=ee_accel_sq,
        )

    def max_reward(self) -> float:
        """Get maximum possible reward (perfect tracking, zero penalties).

        Returns:
            Maximum reward value

        Note:
            For tangent progress, max is approached asymptotically (tanh -> 1).
            At path speed (progress=1.0), r_vel = w_vel * tanh(1) ≈ 0.76 * w_vel.
        """
        return self.w_pos + self.w_ori + self.w_vel

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrackingReward(w_pos={self.w_pos}, w_ori={self.w_ori}, "
            f"w_vel={self.w_vel}, w_action_rate={self.w_action_rate}, "
            f"w_joint_vel={self.w_joint_vel})"
        )
