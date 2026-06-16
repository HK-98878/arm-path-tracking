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
from src.paths.path_factory import create_bspline_path
from src.rl.ppo import PPO
from src.utils.config import load_config
from src.utils.normalization import RunningMeanStd
from src.utils.metrics import compute_jitter_metrics, compute_tracking_error_metrics, compute_ee_jerk_metrics
from src.utils.kinematics import geodesic_angle, rotation_error_rotvec
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
    parser.add_argument('--p-control', action='store_true',
                       help='Use proportional EE-to-target feedback (no RL policy)')
    parser.add_argument('--p-control-gain', type=float, default=1.0,
                       help='P-controller gain (default 1.0 = correct full error per step)')
    parser.add_argument('--p-control-ori-gain', type=float, default=0.0,
                       help='Orientation P-controller gain (fraction of angular error closed per step). '
                            '0.0 = no orientation control (default). Only active with --p-control.')
    parser.add_argument('--reverse-path', action='store_true',
                       help='Reverse path direction (negative speed)')
    parser.add_argument('--fixed-start', action='store_true',
                        help='Force fixed nominal start position regardless of config')
    parser.add_argument('--path-type', type=str, default='circle',
                       choices=['circle', 'figure8', 'bspline'],
                       help='Path type to visualize')
    parser.add_argument('--bspline-seed', type=int, default=None,
                       help='RNG seed for bspline path generation. Omit for a random spline each episode.')
    parser.add_argument('--noise', action='store_true',
                       help='Inject observation noise (from final curriculum stage config). '
                            'Errors if no noise is configured.')
    parser.add_argument('--orientation-mode', type=str, default=None,
                       choices=['fixed', 'rock_x', 'rock_y', 'random_fixed'],
                       help='Force a single orientation mode for the episode (overrides the '
                            'curriculum mix). Omit to sample from the final curriculum stage\'s '
                            'orientation_modes each episode, matching training.')
    parser.add_argument('--render-width', type=int, default=640,
                       help='Render width')
    parser.add_argument('--render-height', type=int, default=480,
                       help='Render height')

    return parser.parse_args()


def setup_mujoco_gl(mode):
    """Print MUJOCO_GL info (already set before imports)."""
    gl_backend = os.environ.get('MUJOCO_GL', 'unknown')
    print(f"Using MUJOCO_GL={gl_backend}")


def get_noise_config(config):
    """Return noise config dict from top-level config or final curriculum stage, or None."""
    noise_section = getattr(config, 'noise', None)
    if noise_section is not None:
        cfg = noise_section.to_dict() if hasattr(noise_section, 'to_dict') else dict(noise_section)
        if any(v > 0 for v in cfg.values()):
            return cfg
    curriculum = getattr(config, 'curriculum', None)
    if curriculum is not None:
        stages = getattr(curriculum, 'stages', None)
        if stages:
            last = stages[-1]
            last_dict = last.to_dict() if hasattr(last, 'to_dict') else dict(last)
            cfg = {
                'obs_noise_pos_std': last_dict.get('obs_noise_pos_std', 0.0),
                'obs_noise_vel_std': last_dict.get('obs_noise_vel_std', 0.0),
                'lookahead_noise_std': last_dict.get('lookahead_noise_std', 0.0),
            }
            if any(v > 0 for v in cfg.values()):
                return cfg
    return None


def _get_final_stage_params(config):
    """Return final curriculum stage as a dict, or {} if curriculum is not enabled."""
    curriculum = getattr(config, 'curriculum', None)
    if curriculum is None or not getattr(curriculum, 'enabled', False):
        return {}
    stages = getattr(curriculum, 'stages', [])
    return dict(stages[-1]) if stages else {}


def make_env(config, render_mode='rgb_array', reverse_path=False, path_type='circle',
             fixed_start=False, bspline_seed=None, obs_noise_config=None,
             orientation_mode_override=None):
    """Create environment from config."""
    # Use the final curriculum stage speed so the visualizer matches training-eval conditions.
    # config.path.speed is the stage-0 base (0.1 m/s); the trained policy runs at stage-5
    # speed (0.2 m/s). Running at the wrong speed produces a 2x mismatch in dynamics.
    final_stage = _get_final_stage_params(config)
    speed = final_stage.get('speed', config.path.speed)
    if reverse_path:
        speed = -speed
        print(f"  Reversed path direction: speed = {speed}")

    # Get orientation variation parameters. Mirror the speed override above: the final
    # curriculum stage's orientation_modes (e.g. ['rock_x', 'rock_y', 'random_fixed']) is
    # what the trained policy actually saw, while config.path.orientation_modes is just the
    # stage-0 default ('fixed'). Using the latter would silently visualize fixed-orientation
    # episodes even for policies trained with orientation variation.
    orientation_modes = final_stage.get(
        'orientation_modes', getattr(config.path, 'orientation_modes', ['fixed'])
    )
    if orientation_mode_override is not None:
        orientation_modes = [orientation_mode_override]
        print(f"  Orientation mode forced: {orientation_mode_override}")
    else:
        print(f"  Orientation modes: {orientation_modes}")
    rock_amplitude = getattr(config.path, 'rock_amplitude', 0.175)
    n_oscillations = getattr(config.path, 'n_oscillations', 2)

    # Enable orientation if w_ori > 0
    include_orientation = getattr(config.reward, 'w_ori', 0) > 0

    if path_type == 'bspline':
        bspline_cfg_obj = getattr(config, 'bspline_path', None)
        bspline_cfg = bspline_cfg_obj.to_dict() if bspline_cfg_obj is not None else {}
        seed_for_init = bspline_seed if bspline_seed is not None else 0
        path = create_bspline_path(
            center=np.array(config.path.center),
            speed=speed,
            bspline_config=bspline_cfg,
            rng=np.random.default_rng(seed_for_init),
            orientation_modes=orientation_modes,
        )
        print(f"  Path: bspline (seed={'fixed:' + str(bspline_seed) if bspline_seed is not None else 'random per episode'})")
    else:
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
        lookahead_n=getattr(config.env, 'lookahead_n', 5),
        lookahead_ds=getattr(config.env, 'lookahead_ds', 0.02),
        curvature_n=getattr(config.env, 'curvature_n', 0),
        randomize_start_position=False if fixed_start else getattr(config.env, 'randomize_start_position', False),
        start_position_noise=getattr(config.env, 'start_position_noise', 0.06),
        p_control_alpha=getattr(config.env, 'p_control_alpha', 0.0),
        p_ori_alpha=getattr(config.env, 'p_ori_alpha', 0.0),
        warmup_steps=getattr(config.env, 'warmup_steps', 0),
        obs_noise_config=obs_noise_config,
    )

    if path_type == 'bspline':
        bspline_cfg_obj = getattr(config, 'bspline_path', None)
        bspline_cfg = bspline_cfg_obj.to_dict() if bspline_cfg_obj is not None else {}
        bspline_override = {}
        if 'min_curvature_radius' in final_stage:
            bspline_override['min_curvature_radius'] = final_stage['min_curvature_radius']
        env._bspline_config = {
            **bspline_cfg,
            **bspline_override,
            'center': np.array(config.path.center),
            'speed': speed,
            'orientation_modes': orientation_modes,
            'rock_amplitude': rock_amplitude,
            'n_oscillations': n_oscillations,
        }
        # Pin the RNG seed so every episode generates the same spline sequence,
        # or leave it unset for a fresh random spline each episode.
        if bspline_seed is not None:
            env.np_random = np.random.default_rng(bspline_seed)

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
        hidden_sizes=tuple(getattr(config.ppo, 'hidden_sizes', [256, 256])),
        network_type=getattr(config.ppo, 'network_type', 'mlp'),
        lstm_hidden_size=getattr(config.ppo, 'lstm_hidden_size', 256),
        seq_len=getattr(config.ppo, 'seq_len', 16),
        device=device,
        caps_config=config.caps.to_dict() if hasattr(config, 'caps') else None
    )

    return agent


def compute_p_control_action(state, target_pos, action_scale, gain,
                             pos_noise_std=0.0, rng=None,
                             target_quat=None, ori_gain=0.0):
    """Proportional feedback: correct EE-to-path-point delta.

    Error is computed in world frame then rotated into EE frame to match
    the action convention (policy outputs are EE-frame residuals).
    pos_noise_std: Gaussian noise on position measurement, matching obs_noise_pos_std seen by RL policy.
    target_quat: Target orientation quaternion [x,y,z,w]; if provided and ori_gain>0, adds angular P-control.
    ori_gain: Proportional gain for orientation correction (fraction of error closed per step).
    """
    ee_pos = state.ee_pos_world + rng.normal(0, pos_noise_std, 3) if pos_noise_std > 0 and rng is not None else state.ee_pos_world
    error_world = target_pos - ee_pos                   # (3,) world frame
    error_ee = state.ee_rot_world.T @ error_world       # EE frame
    action = np.zeros(6, dtype=np.float32)
    action[:3] = gain * error_ee / action_scale[:3]
    if target_quat is not None and ori_gain > 0.0:
        ori_err_world = rotation_error_rotvec(state.ee_quat_world, target_quat)  # world frame
        ori_err_ee = state.ee_rot_world.T @ ori_err_world                        # EE frame
        action[3:] = ori_gain * ori_err_ee / action_scale[3:]
    return np.clip(action, -1.0, 1.0)


def run_episode(env, agent, obs_rms, video_recorder=None,
                episode_recorder=None, deterministic=True,
                feedforward_only=False, p_control=False, p_control_gain=1.0,
                p_control_pos_noise_std=0.0, p_control_ori_gain=0.0):
    """Run single episode with optional recording.

    Returns:
        episode_data: Dict if episode_recorder provided, else None
    """
    obs, _ = env.reset()
    if agent is not None:
        agent.reset_hidden_state()
    obs = obs_rms.normalize(obs)
    done = False
    _pctrl_rng = np.random.default_rng() if p_control_pos_noise_std > 0 else None

    # Sampled per-episode (e.g. one of rock_x/rock_y/random_fixed) — capture for reporting,
    # since env.step()'s info dict doesn't carry it (only reset()'s does).
    orientation_mode = getattr(env.path, '_orientation_mode', 'fixed')
    print(f"  Orientation mode (this episode): {orientation_mode}")

    if episode_recorder:
        episode_recorder.reset()

    step = 0
    while not done:
        # Get current state (needed for P-control and recording)
        state = env._get_robot_state()
        s_current = env.s_current
        s_mod = s_current % env.path.total_length
        target_pos = env.path.position(s_mod)
        target_quat = env.path.orientation(s_mod)

        # Select action
        if feedforward_only:
            action = np.zeros(env.action_space.shape[0])
        elif p_control:
            action = compute_p_control_action(
                state, target_pos, env.action_scale, p_control_gain,
                pos_noise_std=p_control_pos_noise_std, rng=_pctrl_rng,
                target_quat=target_quat, ori_gain=p_control_ori_gain,
            )
        else:
            action, _, _ = agent.select_action(obs, deterministic=deterministic)

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
                obs, action, reward, info, state, target_pos, s_current, target_quat=target_quat
            )

        obs = obs_rms.normalize(next_obs)
        step += 1

    print(f"  Episode completed: {step} steps")

    if episode_recorder is None:
        return None
    episode_data = episode_recorder.get_episode_data()
    episode_data['orientation_mode'] = orientation_mode
    return episode_data


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

    # Error timeline (position, plus orientation if quaternions were tracked)
    target_quats = episode_data.get('target_quaternions')
    orientation_errors = None
    if target_quats is not None and len(target_quats) == len(episode_data['ee_quaternions']):
        orientation_errors = np.array([
            geodesic_angle(q_actual, q_target)
            for q_actual, q_target in zip(episode_data['ee_quaternions'], target_quats)
        ])

    error_path = output_dir / f'episode_{episode_idx:03d}_error.png'
    plotter_traj.plot_error_timeline(
        episode_data['position_errors'],
        dt,
        orientation_errors=orientation_errors,
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


def compute_and_print_metrics(episode_data, dt, feedforward_only=False, p_control=False):
    """Compute and print episode metrics."""
    target_quats = episode_data.get('target_quaternions')
    has_orientation = target_quats is not None and len(target_quats) == len(episode_data['ee_quaternions'])

    # Tracking error
    tracking = compute_tracking_error_metrics(
        episode_data['ee_positions'],
        episode_data['target_positions'],
        orientations=episode_data['ee_quaternions'] if has_orientation else None,
        target_orientations=target_quats if has_orientation else None,
    )

    # EE kinematic jerk — always computed so feedforward and on-policy are comparable
    ee_jerk = compute_ee_jerk_metrics(episode_data['ee_positions'], dt)

    print("\n  Metrics:")
    print(f"    Mean position error: {tracking['mean_position_error']*1000:.2f} mm")
    print(f"    Max position error: {tracking['max_position_error']*1000:.2f} mm")
    print(f"    RMS position error: {tracking['rms_position_error']*1000:.2f} mm")
    if has_orientation:
        print(f"    Mean orientation error: {np.degrees(tracking['mean_orientation_error']):.2f} deg")
        print(f"    Max orientation error: {np.degrees(tracking['max_orientation_error']):.2f} deg")
    print(f"    EE integrated squared jerk: {ee_jerk['integrated_squared_jerk']:.6f} m²/s⁵")
    print(f"    EE RMS jerk: {ee_jerk['rms_jerk']:.6f} m/s³")
    print(f"    EE max jerk: {ee_jerk['max_jerk']:.6f} m/s³")

    if not feedforward_only and not p_control:
        # Action-space jitter is policy-specific — useful for diagnosing the RL controller
        jitter = compute_jitter_metrics(episode_data['actions'], dt)
        print(f"    Action jerk (policy): {jitter['integrated_squared_jerk']:.6f}")
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

    # Resolve noise config
    obs_noise_config = None
    if args.noise:
        obs_noise_config = get_noise_config(config)
        if obs_noise_config is None:
            sys.exit(
                "Error: --noise specified but no noise configuration found. "
                "Add obs_noise_pos_std/obs_noise_vel_std/lookahead_noise_std to the final "
                "curriculum stage or a top-level 'noise' section in the config."
            )
        print(f"  Noise enabled: {obs_noise_config}")

    # Create environment
    print("Creating environment...")
    render_mode = 'rgb_array' if args.mode in ['video', 'headless'] else None
    env = make_env(config, render_mode=render_mode, reverse_path=args.reverse_path,
                   path_type=args.path_type, fixed_start=args.fixed_start,
                   bspline_seed=args.bspline_seed, obs_noise_config=obs_noise_config,
                   orientation_mode_override=args.orientation_mode)
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Create agent and load checkpoint (unless feedforward-only)
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    agent = None

    no_policy = args.feedforward_only or args.p_control
    if no_policy:
        mode_label = "FEEDFORWARD-ONLY" if args.feedforward_only else f"P-CONTROL (gain={args.p_control_gain})"
        print(f"\nSkipping checkpoint load ({mode_label} mode)")
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required unless using --feedforward-only or --p-control")
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
    if no_policy:
        print(f"\n*** {mode_label} MODE ***")
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
        pctrl_pos_noise = obs_noise_config.get('obs_noise_pos_std', 0.0) if (args.p_control and obs_noise_config) else 0.0
        run_kwargs = dict(
            deterministic=args.deterministic,
            feedforward_only=args.feedforward_only,
            p_control=args.p_control,
            p_control_gain=args.p_control_gain,
            p_control_pos_noise_std=pctrl_pos_noise,
            p_control_ori_gain=args.p_control_ori_gain,
        )
        if args.mode == 'viewer':
            # Interactive viewer mode
            viewer = InteractiveViewer(env.model, env.data)
            episode_data = viewer.run_episode(
                env, agent, obs_rms,
                episode_recorder=episode_recorder,
                **run_kwargs
            )
        elif args.mode == 'plot':
            plotter_traj = TrajectoryPlotter()
            episode_data = run_episode(
                env, agent, obs_rms,
                episode_recorder=episode_recorder,
                **run_kwargs
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
                **run_kwargs
            )

        # Save video
        if video_recorder:
            video_recorder.save()

        # Process data
        if episode_data is not None:
            # Compute metrics
            compute_and_print_metrics(
                episode_data, env.dt,
                feedforward_only=args.feedforward_only,
                p_control=args.p_control,
            )

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
