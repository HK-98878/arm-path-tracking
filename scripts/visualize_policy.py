#!/usr/bin/env python3
"""Visualize trained RL policy with MuJoCo rendering and metrics."""

import argparse
import os
import sys

# Must set MUJOCO_GL before any MuJoCo imports
# Parse --mode early to determine GL backend
def _get_mode_from_args():
    for i, arg in enumerate(sys.argv):
        if arg == '--mode' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return 'video'  # default

_mode = _get_mode_from_args()
if _mode == 'viewer':
    os.environ['MUJOCO_GL'] = 'glfw'
else:
    os.environ.setdefault('MUJOCO_GL', 'egl')

from pathlib import Path
import numpy as np
import torch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.ee_tracking_env import EETrackingEnv
from src.paths import create_path
from src.rl.ppo import PPO
from src.utils.config import load_config
from src.utils.normalization import RunningMeanStd
from src.utils.metrics import compute_jitter_metrics, compute_tracking_error_metrics
from src.visualization.episode_recorder import EpisodeRecorder
from src.visualization.mujoco_renderer import VideoRecorder, InteractiveViewer
from src.visualization.trajectory_plotter import TrajectoryPlotter
from src.visualization.metrics_plotter import MetricsPlotter
from src.visualization.data_storage import EpisodeDataStorage


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize trained RL policy'
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint .pt file (optional if --feedforward-only)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--mode', type=str, default='video',
                       choices=['viewer', 'video', 'headless','plot'],
                       help='Visualization mode')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to run')
    parser.add_argument('--save-data', action='store_true',
                       help='Save episode data to HDF5')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save matplotlib plots')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: checkpoint dir + /visualizations)')
    parser.add_argument('--video-fps', type=int, default=50,
                       help='Video frame rate')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy (mean action)')
    parser.add_argument('--feedforward-only', action='store_true',
                       help='Use zero actions (pure feedforward, no policy)')
    parser.add_argument('--reverse-path', action='store_true',
                       help='Reverse path direction (negative speed)')
    parser.add_argument('--fixed-start', action='store_true',
                        help='Force fixed nominal start position regardless of config')
    parser.add_argument('--path-type', type=str, default='circle',
                       choices=['circle', 'figure8'],
                       help='Path type to visualize')
    parser.add_argument('--render-width', type=int, default=640,
                       help='Render width')
    parser.add_argument('--render-height', type=int, default=480,
                       help='Render height')

    return parser.parse_args()


def setup_mujoco_gl(mode):
    """Print MUJOCO_GL info (already set before imports)."""
    gl_backend = os.environ.get('MUJOCO_GL', 'unknown')
    print(f"Using MUJOCO_GL={gl_backend}")


def make_env(config, render_mode='rgb_array', reverse_path=False, path_type='circle', fixed_start=False):
    """Create environment from config."""
    speed = config.path.speed
    if reverse_path:
        speed = -speed
        print(f"  Reversed path direction: speed = {speed}")

    # Get orientation variation parameters
    orientation_modes = getattr(config.path, 'orientation_modes', ['fixed'])
    rock_amplitude = getattr(config.path, 'rock_amplitude', 0.175)
    n_oscillations = getattr(config.path, 'n_oscillations', 2)

    path = create_path(
        path_type=path_type,
        center=np.array(config.path.center),
        radius=config.path.radius,
        speed=speed,
        orientation_modes=orientation_modes,
        rock_amplitude=rock_amplitude,
        n_oscillations=n_oscillations,
    )
    print(f"  Path: {path}")

    # Enable orientation if w_ori > 0
    include_orientation = getattr(config.reward, 'w_ori', 0) > 0

    env = EETrackingEnv(
        model_path=str(project_root / config.env.model_path),
        path=path,
        reward_config=config.reward.to_dict(),
        action_scale=np.array(config.control.action_scale),
        dt=config.env.dt,
        max_episode_steps=config.env.max_episode_steps,
        ee_body_name=config.env.ee_body_name,
        render_mode=render_mode,
        dls_config=config.control.dls.to_dict() if hasattr(config.control, 'dls') else None,
        include_orientation=include_orientation,
        lookahead_ds=getattr(config.env, 'lookahead_ds', 0.02),
        randomize_start_position=False if fixed_start else getattr(config.env, 'randomize_start_position', False),
        start_position_noise=getattr(config.env, 'start_position_noise', 0.06),
    )

    return env


def create_agent(config, env, device='cpu'):
    """Create PPO agent from config."""
    agent = PPO(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=config.ppo.learning_rate,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_epsilon=config.ppo.clip_epsilon,
        n_steps=config.ppo.n_steps,
        n_epochs=config.ppo.n_epochs,
        batch_size=config.ppo.batch_size,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        hidden_sizes=tuple(config.ppo.hidden_sizes),
        device=device,
        caps_config=config.caps.to_dict() if hasattr(config, 'caps') else None
    )

    return agent


def run_episode(env, agent, obs_rms, video_recorder=None,
                episode_recorder=None, deterministic=True, feedforward_only=False):
    """Run single episode with optional recording.

    Returns:
        episode_data: Dict if episode_recorder provided, else None
    """
    obs, _ = env.reset()
    obs = obs_rms.normalize(obs)
    done = False

    if episode_recorder:
        episode_recorder.reset()

    step = 0
    while not done:
        # Select action
        if feedforward_only:
            action = np.zeros(env.action_space.shape[0])
        else:
            action, _, _ = agent.select_action(obs, deterministic=deterministic)

        # Get current state for recording
        state = env._get_robot_state()
        target_pos = env.path.position(env.s_current)
        s_current = env.s_current

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render
        if video_recorder:
            frame = env.render()
            if frame is not None:
                video_recorder.capture_frame(frame)

        # Record data
        if episode_recorder:
            episode_recorder.record_step(
                obs, action, reward, info, state, target_pos, s_current
            )

        obs = obs_rms.normalize(next_obs)
        step += 1

    print(f"  Episode completed: {step} steps")

    return episode_recorder.get_episode_data() if episode_recorder else None


def create_plots(episode_data, output_dir, episode_idx, dt):
    """Generate all plots for an episode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter_traj = TrajectoryPlotter()
    plotter_metrics = MetricsPlotter()

    # 3D trajectory
    traj_path = output_dir / f'episode_{episode_idx:03d}_trajectory.png'
    plotter_traj.plot_3d_trajectory(
        episode_data['ee_positions'],
        episode_data['target_positions'],
        save_path=str(traj_path),
        show=False
    )

    # Error timeline
    error_path = output_dir / f'episode_{episode_idx:03d}_error.png'
    plotter_traj.plot_error_timeline(
        episode_data['position_errors'],
        dt,
        save_path=str(error_path),
        show=False
    )

    # Actions
    actions_path = output_dir / f'episode_{episode_idx:03d}_actions.png'
    plotter_metrics.plot_actions(
        episode_data['actions'],
        dt,
        save_path=str(actions_path),
        show=False
    )

    # Rewards
    rewards_path = output_dir / f'episode_{episode_idx:03d}_rewards.png'
    plotter_metrics.plot_rewards(
        episode_data['rewards'],
        episode_data['reward_components'],
        dt,
        save_path=str(rewards_path),
        show=False
    )

    print(f"  Plots saved to {output_dir}/")


def compute_and_print_metrics(episode_data, dt):
    """Compute and print episode metrics."""
    # Jitter
    jitter = compute_jitter_metrics(episode_data['actions'], dt)

    # Tracking error
    tracking = compute_tracking_error_metrics(
        episode_data['ee_positions'],
        episode_data['target_positions']
    )

    print("\n  Metrics:")
    print(f"    Mean position error: {tracking['mean_position_error']*1000:.2f} mm")
    print(f"    Max position error: {tracking['max_position_error']*1000:.2f} mm")
    print(f"    RMS position error: {tracking['rms_position_error']*1000:.2f} mm")
    print(f"    Integrated squared jerk: {jitter['integrated_squared_jerk']:.6f}")
    print(f"    High-freq power ratio: {jitter['high_freq_power_ratio']:.4f}")
    print(f"    Total reward: {episode_data['rewards'].sum():.2f}")


def main():
    args = parse_args()

    # Setup GL backend before importing mujoco
    setup_mujoco_gl(args.mode)

    # Determine output directory
    if args.output_dir is None:
        if args.checkpoint:
            checkpoint_dir = Path(args.checkpoint).parent
            args.output_dir = checkpoint_dir / 'visualizations'
        else:
            args.output_dir = Path('outputs/feedforward_test')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load config
    print(f"\nLoading config: {args.config}")
    config = load_config(args.config)

    # Create environment
    print("Creating environment...")
    render_mode = 'rgb_array' if args.mode in ['video', 'headless'] else None
    env = make_env(config, render_mode=render_mode, reverse_path=args.reverse_path, path_type=args.path_type, fixed_start=args.fixed_start)
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Create agent and load checkpoint (unless feedforward-only)
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    agent = None

    if args.feedforward_only:
        print("\nSkipping checkpoint load (feedforward-only mode)")
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required unless using --feedforward-only")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        print("\nLoading checkpoint...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = create_agent(config, env, device=device)
        agent.load(args.checkpoint, obs_rms=obs_rms)
        print(f"  Loaded: {args.checkpoint}")
        print(f"  Timesteps trained: {agent.num_timesteps:,}")

    # Initialize components
    data_storage = EpisodeDataStorage() if args.save_data else None

    # Run episodes
    if args.feedforward_only:
        print(f"\n*** FEEDFORWARD-ONLY MODE: Using zero actions (no policy) ***")
    print(f"\nRunning {args.episodes} episode(s)...")
    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}")

        # Video recorder
        video_recorder = None
        if args.mode == 'video':
            video_path = output_dir / f'episode_{ep:03d}.mp4'
            video_recorder = VideoRecorder(
                str(video_path),
                fps=args.video_fps,
                codec='h264',
                quality=8
            )

        # Episode recorder
        episode_recorder = EpisodeRecorder() if (args.save_data or args.save_plots or args.mode == 'plot') else None

        # Run episode
        if args.mode == 'viewer':
            # Interactive viewer mode
            viewer = InteractiveViewer(env.model, env.data)
            episode_data = viewer.run_episode(
                env, agent, obs_rms,
                episode_recorder=episode_recorder,
                deterministic=args.deterministic,
                feedforward_only=args.feedforward_only
            )
        elif args.mode == 'plot':
            plotter_traj = TrajectoryPlotter()
            episode_data = run_episode(
                env, agent, obs_rms,
                episode_recorder=episode_recorder,
                deterministic=args.deterministic,
                feedforward_only=args.feedforward_only
            )
            # Interactive playback with slider
            plotter_traj.plot_3d_playback(
                episode_data['ee_positions'],  # type: ignore
                episode_data['target_positions'],  # type: ignore
                dt=env.dt
            )
        else:
            # Video or headless mode
            episode_data = run_episode(
                env, agent, obs_rms,
                video_recorder=video_recorder,
                episode_recorder=episode_recorder,
                deterministic=args.deterministic,
                feedforward_only=args.feedforward_only
            )

        # Save video
        if video_recorder:
            video_recorder.save()

        # Process data
        if episode_data is not None:
            # Compute metrics
            compute_and_print_metrics(episode_data, env.dt)

            # Generate plots
            if args.save_plots:
                create_plots(episode_data, output_dir, ep, env.dt)

            # Save data
            if args.save_data:
                data_path = output_dir / f'episode_{ep:03d}.h5'
                metadata = {
                    'checkpoint_path': args.checkpoint,
                    'config_path': args.config,
                    'episode_idx': ep,
                    'dt': env.dt,
                    'deterministic': args.deterministic,
                }
                data_storage.save(episode_data, str(data_path), metadata)

    env.close()
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Outputs saved to: {output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
