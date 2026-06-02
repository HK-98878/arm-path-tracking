#!/usr/bin/env python3
"""Analyze trajectory to find position-dependent error patterns.

Plots error metrics vs arc length to identify problematic regions.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.ee_tracking_env import EETrackingEnv
from src.paths.circle_path import CirclePath
from src.rl.ppo import PPO
from src.utils.config import load_config
from src.utils.normalization import RunningMeanStd
from src.utils.kinematics import geodesic_angle
from src.utils.manipulability import compute_manipulability, joint_limit_proximity
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze trajectory errors vs position')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config')
    parser.add_argument('--output', type=str, default=None, help='Output plot path')
    parser.add_argument('--feedforward-only', action='store_true', help='Use zero actions')
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    # Create environment
    path = CirclePath(
        radius=config.path.radius,
        center=np.array(config.path.center),
        speed=config.path.speed
    )

    env = EETrackingEnv(
        model_path=str(project_root / config.env.model_path),
        path=path,
        reward_config=config.reward.to_dict(),
        action_scale=np.array(config.control.action_scale),
        dt=config.env.dt,
        max_episode_steps=config.env.max_episode_steps,
        ee_body_name=config.env.ee_body_name,
        render_mode=None,
        dls_config=config.control.dls.to_dict() if hasattr(config.control, 'dls') else None,
        include_orientation=True
    )

    # Load agent and normalization
    agent = None
    if not args.feedforward_only:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        agent = PPO(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            learning_rate=config.ppo.learning_rate,
            hidden_sizes=tuple(config.ppo.hidden_sizes),
            device='cpu'
        )
        agent.actor_critic.load_state_dict(checkpoint['actor_critic'])
        agent.actor_critic.eval()

    obs_rms = RunningMeanStd(shape=(env.observation_space.shape[0],))
    if agent is None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'obs_rms_mean' in checkpoint:
        obs_rms.mean = checkpoint['obs_rms_mean']
        obs_rms.var = checkpoint['obs_rms_var']
        obs_rms.count = checkpoint.get('obs_rms_count', 1000)

    # Run episode and collect data
    obs, _ = env.reset()
    obs = obs_rms.normalize(obs)

    data = {
        's': [],
        'pos_error': [],
        'ori_error': [],
        'z_error': [],
        'manipulability': [],
        'joint_limit_min': [],
        'ee_x': [],
        'ee_y': [],
        'ee_z': [],
    }

    done = False
    while not done:
        # Select action
        if args.feedforward_only:
            action = np.zeros(env.action_space.shape[0])
        else:
            action, _, _ = agent.select_action(obs, deterministic=True)

        # Get state before step
        state = env._get_robot_state()
        s_wrapped = env.s_current % env.path.total_length
        target_pos = env.path.position(s_wrapped)
        target_quat = env.path.orientation(s_wrapped)

        # Compute metrics
        pos_error = np.linalg.norm(state.ee_pos_world - target_pos)
        ori_error = geodesic_angle(state.ee_quat_world, target_quat)
        z_error = state.ee_pos_world[2] - target_pos[2]
        manip = compute_manipulability(state.jacobian)
        limit_prox = joint_limit_proximity(state.joint_pos, (env.q_min, env.q_max))

        # Store
        data['s'].append(s_wrapped)
        data['pos_error'].append(pos_error * 1000)  # mm
        data['ori_error'].append(np.degrees(ori_error))  # degrees
        data['z_error'].append(z_error * 1000)  # mm
        data['manipulability'].append(manip)
        data['joint_limit_min'].append(np.min(limit_prox))
        data['ee_x'].append(state.ee_pos_world[0])
        data['ee_y'].append(state.ee_pos_world[1])
        data['ee_z'].append(state.ee_pos_world[2])

        # Step
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = obs_rms.normalize(next_obs)

    # Convert to arrays
    for k in data:
        data[k] = np.array(data[k])

    # Sort by arc length for cleaner plots
    sort_idx = np.argsort(data['s'])
    for k in data:
        data[k] = data[k][sort_idx]

    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Trajectory Analysis vs Arc Length', fontsize=14)

    # Position error vs s
    ax = axes[0, 0]
    ax.plot(data['s'], data['pos_error'], 'b-', linewidth=0.8)
    ax.set_xlabel('Arc length s (m)')
    ax.set_ylabel('Position error (mm)')
    ax.set_title('Position Error')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.mean(data['pos_error']), color='r', linestyle='--', label=f"mean={np.mean(data['pos_error']):.1f}mm")
    ax.legend()

    # Orientation error vs s
    ax = axes[0, 1]
    ax.plot(data['s'], data['ori_error'], 'g-', linewidth=0.8)
    ax.set_xlabel('Arc length s (m)')
    ax.set_ylabel('Orientation error (deg)')
    ax.set_title('Orientation Error')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.mean(data['ori_error']), color='r', linestyle='--', label=f"mean={np.mean(data['ori_error']):.1f}°")
    ax.legend()

    # Z error vs s
    ax = axes[1, 0]
    ax.plot(data['s'], data['z_error'], 'm-', linewidth=0.8)
    ax.set_xlabel('Arc length s (m)')
    ax.set_ylabel('Z error (mm)')
    ax.set_title('Z-axis Error (signed)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Manipulability vs s
    ax = axes[1, 1]
    ax.plot(data['s'], data['manipulability'], 'c-', linewidth=0.8)
    ax.set_xlabel('Arc length s (m)')
    ax.set_ylabel('Manipulability')
    ax.set_title('Manipulability (higher = better conditioned)')
    ax.grid(True, alpha=0.3)

    # Joint limit proximity vs s
    ax = axes[2, 0]
    ax.plot(data['s'], data['joint_limit_min'], 'orange', linewidth=0.8)
    ax.set_xlabel('Arc length s (m)')
    ax.set_ylabel('Min joint limit proximity')
    ax.set_title('Joint Limit Proximity (0=at limit, 1=centered)')
    ax.grid(True, alpha=0.3)

    # XY trajectory colored by orientation error
    ax = axes[2, 1]
    scatter = ax.scatter(data['ee_y'], data['ee_x'], c=data['ori_error'],
                         cmap='hot', s=2, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Ori error (deg)')
    # Draw target circle
    theta = np.linspace(0, 2*np.pi, 100)
    cx, cy = config.path.center[0], config.path.center[1]
    r = config.path.radius
    ax.plot(cy + r*np.sin(theta), cx + r*np.cos(theta), 'b--', alpha=0.5, label='Target')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    ax.set_title('XY Trajectory (colored by ori error)')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()

    # Print summary
    print("\n=== Summary ===")
    print(f"Position error: {np.mean(data['pos_error']):.2f} ± {np.std(data['pos_error']):.2f} mm")
    print(f"Orientation error: {np.mean(data['ori_error']):.2f} ± {np.std(data['ori_error']):.2f} deg")
    print(f"Manipulability: {np.mean(data['manipulability']):.4f} (min={np.min(data['manipulability']):.4f})")
    print(f"Joint limit proximity min: {np.min(data['joint_limit_min']):.3f}")

    # Find worst regions
    worst_pos_idx = np.argmax(data['pos_error'])
    worst_ori_idx = np.argmax(data['ori_error'])
    print(f"\nWorst position error: {data['pos_error'][worst_pos_idx]:.1f}mm at s={data['s'][worst_pos_idx]:.3f}m")
    print(f"Worst orientation error: {data['ori_error'][worst_ori_idx]:.1f}° at s={data['s'][worst_ori_idx]:.3f}m")


if __name__ == "__main__":
    main()
