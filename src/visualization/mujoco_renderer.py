"""MuJoCo rendering utilities for video export and interactive viewing."""

import numpy as np
import imageio
import mujoco
import mujoco.viewer
import time


class VideoRecorder:
    """Record episode to MP4 video file."""

    def __init__(self, filename, fps=50, codec='h264', quality=8):
        """Initialize video recorder.

        Args:
            filename: Output path (e.g., 'episode.mp4')
            fps: Frames per second
            codec: Video codec ('h264' recommended)
            quality: 0-10, higher = better (8 = good default)
        """
        self.filename = filename
        self.fps = fps
        self.writer = imageio.get_writer(
            filename,
            fps=fps,
            codec=codec,
            quality=quality
        )
        self.frames = []

    def capture_frame(self, frame_rgb):
        """Add frame to buffer.

        Args:
            frame_rgb: (H, W, 3) uint8 array
        """
        if frame_rgb is not None:
            self.frames.append(frame_rgb)

    def save(self):
        """Write all frames to video file."""
        for frame in self.frames:
            self.writer.append_data(frame)
        self.writer.close()
        print(f"Video saved: {self.filename} ({len(self.frames)} frames)")


class InteractiveViewer:
    """Interactive MuJoCo viewer for real-time policy visualization."""

    def __init__(self, model, data):
        """Initialize interactive viewer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data

    def run_episode(self, env, agent, obs_rms, episode_recorder=None,
                    deterministic=True, feedforward_only=False,
                    p_control=False, p_control_gain=1.0):
        """Run episode with interactive viewer.

        Args:
            env: Gymnasium environment
            agent: PPO agent (can be None if feedforward_only/p_control)
            obs_rms: Observation normalization stats
            episode_recorder: Optional EpisodeRecorder for data collection
            deterministic: Use deterministic policy
            feedforward_only: Use zero actions (pure feedforward, no policy)
            p_control: Use proportional EE-to-target feedback
            p_control_gain: Gain for P-controller

        Returns:
            episode_data: Dict if episode_recorder provided, else None
        """
        obs, _ = env.reset()
        obs = obs_rms.normalize(obs)
        done = False

        if episode_recorder:
            episode_recorder.reset()

        step = 0

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Configure initial camera view (use env's camera if available)
            if hasattr(env, 'camera') and env.camera is not None:
                viewer.cam.lookat[:] = env.camera.lookat
                viewer.cam.distance = env.camera.distance
                viewer.cam.azimuth = env.camera.azimuth
                viewer.cam.elevation = env.camera.elevation

            while viewer.is_running() and not done:
                step_start = time.time()

                state = env._get_robot_state()
                target_pos = env.path.position(env.s_current % env.path.total_length)

                # Select action
                if feedforward_only:
                    action = np.zeros(env.action_space.shape[0])
                elif p_control:
                    error_world = target_pos - state.ee_pos_world
                    error_ee = state.ee_rot_world.T @ error_world
                    action = np.zeros(env.action_space.shape[0], dtype=np.float32)
                    action[:3] = p_control_gain * error_ee / env.action_scale[:3]
                    action = np.clip(action, -1.0, 1.0)
                else:
                    action, _, _ = agent.select_action(obs, deterministic=deterministic)

                s_current = env.s_current

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Record data
                if episode_recorder:
                    episode_recorder.record_step(
                        obs, action, reward, info, state, target_pos, s_current
                    )

                # Sync viewer
                viewer.sync()

                obs = obs_rms.normalize(next_obs)
                step += 1

                # Maintain real-time playback (respect dt)
                elapsed = time.time() - step_start
                if elapsed < env.dt:
                    time.sleep(env.dt - elapsed)

        print(f"  Episode completed: {step} steps")

        return episode_recorder.get_episode_data() if episode_recorder else None
