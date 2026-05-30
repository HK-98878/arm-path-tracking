"""Training script for Step 1: Circle baseline (position tracking only).

This script implements the first step of the progressive build order:
- Circle path tracking
- Position-only reward
- Standard PPO (no CAPS)
- DLS controller
- Headless training
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.ee_tracking_env import EETrackingEnv
from src.paths.circle_path import CirclePath
from src.rl.ppo import PPO
from src.utils.config import load_config
from src.utils.metrics import compute_jitter_metrics, compute_tracking_error_metrics
from src.utils.normalization import RunningMeanStd


def make_env(config):
    """Create environment from config."""
    # Create path
    path = CirclePath(
        radius=config.path.radius,
        center=np.array(config.path.center),
        speed=config.path.speed
    )

    # Create environment
    env = EETrackingEnv(
        model_path=str(project_root / config.env.model_path),
        path=path,
        reward_config=config.reward.to_dict(),
        action_scale=np.array(config.control.action_scale),
        dt=config.env.dt,
        max_episode_steps=config.env.max_episode_steps,
        ee_body_name=config.env.ee_body_name,
        render_mode=None,  # Headless
        dls_config=config.control.dls.to_dict() if hasattr(config.control, 'dls') else None
    )

    return env


def evaluate(env, agent, obs_rms, num_episodes=5):
    """Evaluate agent performance.

    Args:
        env: Environment
        agent: PPO agent
        obs_rms: Observation normalization statistics
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    all_actions = []
    all_positions = []
    all_targets = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs = obs_rms.normalize(obs)  # Normalize observation
        done = False
        episode_reward = 0
        episode_length = 0
        episode_actions = []
        episode_positions = []
        episode_targets = []

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)

            # Store for metrics
            episode_actions.append(action)

            # Get current state for tracking
            state = env._get_robot_state()
            target_pos = env.path.position(env.s_current)
            episode_positions.append(state.ee_pos_world)
            episode_targets.append(target_pos)

            obs, reward, terminated, truncated, info = env.step(action)
            obs = obs_rms.normalize(obs)  # Normalize observation
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_actions.append(np.array(episode_actions))
        all_positions.append(np.array(episode_positions))
        all_targets.append(np.array(episode_targets))

    # Aggregate metrics
    metrics = {
        'mean_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
    }

    # Jitter metrics (from first episode)
    if len(all_actions) > 0:
        jitter = compute_jitter_metrics(all_actions[0], env.dt)
        metrics.update({f'jitter_{k}': v for k, v in jitter.items()})

    # Tracking error metrics (from first episode)
    if len(all_positions) > 0 and len(all_targets) > 0:
        tracking = compute_tracking_error_metrics(
            all_positions[0],
            all_targets[0]
        )
        metrics.update({f'tracking_{k}': v for k, v in tracking.items()})

    return metrics


def train(config):
    """Main training loop.

    Args:
        config: Configuration object
    """
    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Set MuJoCo to headless mode
    os.environ['MUJOCO_GL'] = 'egl'

    # Create output directory
    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    print("Creating environment...")
    env = make_env(config)
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Create agent
    print("\nCreating PPO agent...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

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

    print(f"\nTraining for {config.training.total_timesteps:,} timesteps...")
    print(f"  n_steps per update: {config.ppo.n_steps}")
    print(f"  Total updates: {config.training.total_timesteps // config.ppo.n_steps}")

    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    print(f"  Using running observation normalization")

    # Training loop
    obs, _ = env.reset()
    obs_rms.update(obs)
    obs = obs_rms.normalize(obs)
    episode_reward = 0
    episode_length = 0
    num_episodes = 0

    for timestep in range(config.training.total_timesteps):
        # Select action
        action, value, log_prob = agent.select_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        obs_rms.update(next_obs)
        next_obs_normalized = obs_rms.normalize(next_obs)

        # Store transition
        agent.store_transition(obs, action, reward, value, log_prob, done)

        episode_reward += reward
        episode_length += 1

        # Update observation
        obs = next_obs_normalized

        # Episode end
        if done:
            num_episodes += 1
            obs, _ = env.reset()
            obs_rms.update(obs)
            obs = obs_rms.normalize(obs)
            episode_reward = 0
            episode_length = 0

        # Update policy
        if (timestep + 1) % config.ppo.n_steps == 0:
            update_metrics = agent.update(obs, done)

            # Log training metrics
            if (timestep + 1) % config.training.log_frequency == 0:
                print(f"\nTimestep {timestep + 1:,} / {config.training.total_timesteps:,}")
                print(f"  Episodes: {num_episodes}")
                print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
                print(f"  Value loss: {update_metrics['value_loss']:.4f}")
                print(f"  Entropy: {-update_metrics['entropy_loss']:.4f}")
                print(f"  Approx KL: {update_metrics['approx_kl']:.4f}")
                print(f"  Clip fraction: {update_metrics['clip_fraction']:.4f}")

        # Evaluation
        if (timestep + 1) % config.training.eval_frequency == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at timestep {timestep + 1:,}")
            print('='*60)

            eval_metrics = evaluate(env, agent, obs_rms, config.training.num_eval_episodes)

            print(f"  Mean episode reward: {eval_metrics['mean_episode_reward']:.2f} ± {eval_metrics['std_episode_reward']:.2f}")
            print(f"  Mean episode length: {eval_metrics['mean_episode_length']:.1f}")
            print(f"  Mean position error: {eval_metrics['tracking_mean_position_error']*1000:.2f} mm")
            print(f"  Max position error: {eval_metrics['tracking_max_position_error']*1000:.2f} mm")
            print(f"  Jitter (integrated squared jerk): {eval_metrics['jitter_integrated_squared_jerk']:.6f}")
            print(f"  High-freq power ratio: {eval_metrics['jitter_high_freq_power_ratio']:.4f}")

        # Save checkpoint
        if (timestep + 1) % config.training.save_frequency == 0:
            checkpoint_path = output_dir / f"checkpoint_{timestep + 1}.pt"
            agent.save(str(checkpoint_path), obs_rms=obs_rms)
            print(f"\n  Saved checkpoint: {checkpoint_path}")

    # Final save
    final_path = output_dir / "final_model.pt"
    agent.save(str(final_path), obs_rms=obs_rms)
    print(f"\n{'='*60}")
    print(f"Training complete! Final model saved to: {final_path}")
    print('='*60)

    env.close()


if __name__ == "__main__":
    # Load configuration
    config_path = project_root / "configs" / "circle_baseline.yaml"
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Train
    train(config)
