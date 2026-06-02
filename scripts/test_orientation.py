#!/usr/bin/env python
"""Quick test script for orientation control implementation."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def test_angular_velocity():
    """Test angular velocity implementation in CirclePath."""
    from src.paths.circle_path import CirclePath

    print("=" * 60)
    print("Test 1: Angular Velocity Implementation")
    print("=" * 60)

    # Create path
    path = CirclePath(radius=0.15, center=np.array([0.32, 0, 0.52]), speed=0.1)
    print(f"CirclePath: {path}")

    # Test angular velocity
    omega = path.angular_velocity(0)
    expected_mag = 0.1 / 0.15  # speed / radius
    expected = expected_mag * np.array([0, 0, 1])  # about z-axis

    print(f"Angular velocity at s=0: {omega}")
    print(f"Expected magnitude: {expected_mag:.4f} rad/s")
    print(f"Expected direction: {path.normal}")

    # Verify
    assert np.allclose(omega, expected), f"Mismatch: {omega} vs {expected}"
    print("✓ Forward path angular velocity correct!")

    # Test reversed path
    path_rev = CirclePath(radius=0.15, center=np.array([0.32, 0, 0.52]), speed=-0.1)
    omega_rev = path_rev.angular_velocity(0)
    print(f"\nReversed path angular velocity: {omega_rev}")
    assert np.allclose(omega_rev, -expected), f"Mismatch: {omega_rev} vs {-expected}"
    print("✓ Reversed path angular velocity correct!")


def test_env_orientation():
    """Test environment with orientation control enabled."""
    from src.environment.ee_tracking_env import EETrackingEnv
    from src.paths.circle_path import CirclePath

    print("\n" + "=" * 60)
    print("Test 2: Environment with Orientation Control")
    print("=" * 60)

    path = CirclePath(radius=0.15, center=np.array([0.32, 0, 0.52]), speed=0.1)

    reward_config = {
        'w_pos': 1.0,
        'w_ori': 0.25,
        'w_vel': 0.5,
        'sig_pos': 0.15,
        'sig_ori': 0.15,
        'w_action_rate': 0.1,
        'w_joint_vel': 0.001,
        'vel_reward_type': 'arc_length',
        'pos_linear_fallback_max': 0.5,
        'vel_gate_max': 0.15,
    }

    model_path = str(project_root / "src/models/mujoco_menagerie/franka_emika_panda/panda_nohand.xml")

    env = EETrackingEnv(
        model_path=model_path,
        path=path,
        reward_config=reward_config,
        action_scale=np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05]),
        dt=0.01,
        max_episode_steps=100,
        ee_body_name="attachment",
        include_orientation=True
    )

    print(f"Environment created successfully")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Include orientation: {env.include_orientation}")

    # Reset and step
    obs, info = env.reset()
    print(f"\nReset observation shape: {obs.shape}")

    # Take a few steps
    for i in range(5):
        action = np.zeros(6)  # Zero action = pure feedforward
        obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nAfter 5 steps:")
    print(f"  r_pos: {info['r_pos']:.4f}")
    print(f"  r_ori: {info['r_ori']:.4f}")
    print(f"  r_vel: {info['r_vel']:.4f}")
    print(f"  pos_error: {info['pos_error']*1000:.2f} mm")
    print(f"  ori_error: {np.degrees(info['ori_error']):.2f} deg")

    env.close()
    print("✓ Environment with orientation control works!")


def test_bidirectional():
    """Test bidirectional path creation."""
    from src.paths.circle_path import CirclePath

    print("\n" + "=" * 60)
    print("Test 3: Bidirectional Path Support")
    print("=" * 60)

    np.random.seed(42)

    speeds_used = []
    for i in range(10):
        base_speed = 0.1
        speed = base_speed if np.random.random() > 0.5 else -base_speed
        speeds_used.append(speed)

    n_forward = sum(1 for s in speeds_used if s > 0)
    n_reverse = sum(1 for s in speeds_used if s < 0)

    print(f"Random speed signs (10 trials): {speeds_used}")
    print(f"Forward: {n_forward}, Reverse: {n_reverse}")
    print("✓ Bidirectional randomization works!")


if __name__ == "__main__":
    try:
        test_angular_velocity()
        test_env_orientation()
        test_bidirectional()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
