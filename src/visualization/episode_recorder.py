"""Episode data recorder for trajectory analysis."""

import numpy as np
from typing import Dict, Any


class EpisodeRecorder:
    """Records all data from a single episode for analysis."""

    def __init__(self):
        """Initialize recorder."""
        self.reset()

    def reset(self):
        """Clear buffers for new episode."""
        self.data = {
            # Policy
            'observations': [],
            'actions': [],
            'rewards': [],
            'reward_components': [],

            # Robot state
            'joint_positions': [],
            'joint_velocities': [],
            'ee_positions': [],
            'ee_quaternions': [],
            'ee_linear_velocities': [],

            # Reference path
            'target_positions': [],
            'target_quaternions': [],
            'arc_lengths': [],

            # Metrics
            'position_errors': [],

            # Metadata
            'timestamps': [],
        }
        self.step_count = 0

    def record_step(self, obs, action, reward, info, state, target_pos, s, target_quat=None):
        """Record one timestep.

        Args:
            obs: Observation (normalized)
            action: Policy action
            reward: Scalar reward
            info: Info dict from env.step()
            state: RobotState object from env._get_robot_state()
            target_pos: Reference position
            s: Arc length
            target_quat: Reference orientation quaternion [x,y,z,w], if tracked
        """
        self.data['observations'].append(obs.copy())
        self.data['actions'].append(action.copy())
        self.data['rewards'].append(reward)
        self.data['reward_components'].append(dict(info))  # Contains reward breakdown

        # State
        self.data['joint_positions'].append(state.joint_pos.copy())
        self.data['joint_velocities'].append(state.joint_vel.copy())
        self.data['ee_positions'].append(state.ee_pos_world.copy())
        self.data['ee_quaternions'].append(state.ee_quat_world.copy())
        self.data['ee_linear_velocities'].append(state.ee_lin_vel_world.copy())

        # Reference
        self.data['target_positions'].append(target_pos.copy())
        if target_quat is not None:
            self.data['target_quaternions'].append(target_quat.copy())
        self.data['arc_lengths'].append(s)

        # Metrics
        self.data['position_errors'].append(info.get('pos_error', 0.0))

        self.data['timestamps'].append(self.step_count)
        self.step_count += 1

    def get_episode_data(self) -> Dict[str, Any]:
        """Return episode data as dict of numpy arrays."""
        return {
            k: np.array(v) if isinstance(v, list) and len(v) > 0 and not isinstance(v[0], dict) else v
            for k, v in self.data.items()
        }
