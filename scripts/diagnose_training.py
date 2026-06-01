"""Diagnostic script to understand training dynamics.

Run this to check:
1. Whether feedforward alone (action=0) tracks the path
2. What magnitudes feedforward vs residual have
3. Whether the policy's exploration is swamping feedforward
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ['MUJOCO_GL'] = 'egl'

from src.environment.ee_tracking_env import EETrackingEnv
from src.paths.circle_path import CirclePath
from src.utils.config import load_config


def test_feedforward_only(env, num_steps=100):
    """Test if pure feedforward (action=0) tracks the path."""
    print("\n" + "="*60)
    print("TEST 1: Pure feedforward (action=0)")
    print("="*60)

    obs, _ = env.reset()

    errors = []
    for step in range(num_steps):
        action = np.zeros(6)  # Pure feedforward
        obs, reward, term, trunc, info = env.step(action)
        errors.append(info['pos_error'])

        if step % 20 == 0:
            print(f"  Step {step:3d}: error = {info['pos_error']*1000:6.1f}mm, reward = {reward:.4f}")

    print(f"\n  Initial error: {errors[0]*1000:.1f}mm")
    print(f"  Final error:   {errors[-1]*1000:.1f}mm")
    print(f"  Mean error:    {np.mean(errors)*1000:.1f}mm")
    print(f"  Max error:     {np.max(errors)*1000:.1f}mm")

    if errors[-1] > errors[0] * 2:
        print("\n  ⚠️  ERROR GROWING: Feedforward alone is NOT tracking!")
        print("     The reference is outrunning the robot.")
    elif errors[-1] < errors[0] * 0.5:
        print("\n  ✓ Feedforward is reducing error - basic tracking works")
    else:
        print("\n  ~ Error roughly stable with feedforward only")

    return errors


def test_random_actions(env, num_steps=100):
    """Test what happens with random actions (simulating untrained policy)."""
    print("\n" + "="*60)
    print("TEST 2: Random actions from N(0,1) -> tanh (untrained policy)")
    print("="*60)

    obs, _ = env.reset()

    errors = []
    action_magnitudes = []

    for step in range(num_steps):
        # Simulate untrained policy: N(0,1) -> tanh
        raw = np.random.randn(6)
        action = np.tanh(raw)
        action_magnitudes.append(np.abs(action))

        obs, reward, term, trunc, info = env.step(action)
        errors.append(info['pos_error'])

        if step % 20 == 0:
            print(f"  Step {step:3d}: error = {info['pos_error']*1000:6.1f}mm, action_mag = {np.mean(np.abs(action)):.3f}")

    print(f"\n  Initial error: {errors[0]*1000:.1f}mm")
    print(f"  Final error:   {errors[-1]*1000:.1f}mm")
    print(f"  Mean error:    {np.mean(errors)*1000:.1f}mm")
    print(f"  Mean |action|: {np.mean(action_magnitudes):.3f}")

    return errors


def analyze_feedforward_vs_residual(env):
    """Compare magnitudes of feedforward vs residual."""
    print("\n" + "="*60)
    print("TEST 3: Feedforward vs Residual magnitudes")
    print("="*60)

    obs, _ = env.reset()

    # Feedforward magnitude
    v_ref = env.path.velocity(env.s_current)
    ff_linear = np.linalg.norm(v_ref * env.dt)
    print(f"\n  Feedforward (per step):")
    print(f"    Linear:  {ff_linear*1000:.3f}mm")
    print(f"    Angular: 0 (Step 1)")

    # Residual with random action
    action_scale = env.action_scale
    print(f"\n  Action scale: {action_scale}")

    # With action ∈ [-1, 1] uniformly
    max_residual_linear = np.linalg.norm(action_scale[:3])
    mean_residual_linear = max_residual_linear * 0.5  # Rough estimate

    print(f"\n  Max residual (action=±1):")
    print(f"    Linear:  {max_residual_linear*1000:.1f}mm")

    print(f"\n  Ratio (max_residual / feedforward): {max_residual_linear / ff_linear:.1f}x")

    if max_residual_linear > ff_linear * 10:
        print("\n  ⚠️  Residual can be >10x feedforward!")
        print("     Random exploration will completely swamp feedforward.")
        print("     Consider: reduce ACTION_SCALE or initialize log_std < 0")


def test_small_std_actions(env, num_steps=100, log_std=-1.0):
    """Test with smaller exploration (log_std < 0)."""
    print("\n" + "="*60)
    print(f"TEST 4: Smaller exploration (log_std={log_std}, std={np.exp(log_std):.3f})")
    print("="*60)

    obs, _ = env.reset()
    std = np.exp(log_std)

    errors = []
    action_magnitudes = []

    for step in range(num_steps):
        # Simulate policy with smaller std
        raw = np.random.randn(6) * std
        action = np.tanh(raw)
        action_magnitudes.append(np.abs(action))

        obs, reward, term, trunc, info = env.step(action)
        errors.append(info['pos_error'])

        if step % 20 == 0:
            print(f"  Step {step:3d}: error = {info['pos_error']*1000:6.1f}mm, action_mag = {np.mean(np.abs(action)):.3f}")

    print(f"\n  Initial error: {errors[0]*1000:.1f}mm")
    print(f"  Final error:   {errors[-1]*1000:.1f}mm")
    print(f"  Mean error:    {np.mean(errors)*1000:.1f}mm")
    print(f"  Mean |action|: {np.mean(action_magnitudes):.3f}")

    return errors


def main():
    config_path = project_root / "configs" / "circle_baseline.yaml"
    print(f"Loading config: {config_path}")
    config = load_config(config_path)

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
        render_mode=None,
        dls_config=config.control.dls.to_dict() if hasattr(config.control, 'dls') else None
    )

    print(f"\nEnvironment created:")
    print(f"  Circle: radius={config.path.radius}m, center={config.path.center}")
    print(f"  Speed: {config.path.speed} m/s")
    print(f"  dt: {config.env.dt}s")

    # Run tests
    test_feedforward_only(env, num_steps=200)
    analyze_feedforward_vs_residual(env)
    test_random_actions(env, num_steps=200)
    test_small_std_actions(env, num_steps=200, log_std=-1.0)
    test_small_std_actions(env, num_steps=200, log_std=-2.0)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
If TEST 1 shows error growing:
  → Feedforward alone can't track. Check DLS controller / actuators.

If TEST 1 works but TEST 2 fails:
  → Random exploration swamps feedforward.
  → Fix: Initialize log_std to -1 or -2 in networks.py

If TEST 4 with log_std=-1 works better than TEST 2:
  → Confirms exploration is the issue.
  → Apply the fix to networks.py
""")

    env.close()


if __name__ == "__main__":
    main()
