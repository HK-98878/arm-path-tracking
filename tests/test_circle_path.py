"""Unit tests for circle path generation."""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paths.circle_path import CirclePath, create_horizontal_circle


def test_circle_geometry():
    """Test that circle path produces correct geometric properties."""
    # Create a simple circle: radius 0.3m, centered at [0.5, 0, 0.3]
    path = create_horizontal_circle(radius=0.3, height=0.3, speed=0.2)

    print("=" * 60)
    print("Testing Circle Path Geometry")
    print("=" * 60)
    print(f"Circle: {path}\n")

    # Test 1: Positions should lie on circle
    print("Test 1: Verify positions lie on circle")
    test_points = [0.0, np.pi * 0.3 / 2, np.pi * 0.3, 2 * np.pi * 0.3]
    for s in test_points:
        pos = path.position(s)
        # Distance from center should equal radius
        dist = np.linalg.norm(pos - path.center)
        print(f"  s={s:.3f}: pos={pos}, dist from center={dist:.6f} (expected {path.radius})")
        assert np.isclose(dist, path.radius, atol=1e-6), \
            f"Position not on circle: dist={dist}, expected={path.radius}"

    # Test 2: Velocity should have constant magnitude (speed)
    print("\nTest 2: Verify velocity magnitude equals speed")
    for s in test_points:
        vel = path.velocity(s)
        speed = np.linalg.norm(vel)
        print(f"  s={s:.3f}: vel={vel}, speed={speed:.6f} (expected {path.speed})")
        assert np.isclose(speed, path.speed, atol=1e-6), \
            f"Incorrect speed: {speed}, expected={path.speed}"

    # Test 3: Velocity should be perpendicular to radius vector
    print("\nTest 3: Verify velocity perpendicular to radius")
    for s in test_points:
        pos = path.position(s)
        vel = path.velocity(s)
        radius_vec = pos - path.center
        dot = np.dot(vel, radius_vec)
        print(f"  s={s:.3f}: vel·radius={dot:.6e} (should be ~0)")
        assert np.abs(dot) < 1e-6, f"Velocity not perpendicular: dot={dot}"

    # Test 4: Constant curvature
    print("\nTest 4: Verify constant curvature")
    expected_curvature = 1.0 / path.radius
    for s in test_points:
        curv = path.curvature(s)
        print(f"  s={s:.3f}: κ={curv:.6f} (expected {expected_curvature:.6f})")
        assert np.isclose(curv, expected_curvature), \
            f"Incorrect curvature: {curv}, expected={expected_curvature}"

    # Test 5: Total arc length
    print(f"\nTest 5: Total arc length = {path.total_length:.6f} m")
    expected_length = 2 * np.pi * path.radius
    assert np.isclose(path.total_length, expected_length), \
        f"Incorrect total length: {path.total_length}, expected={expected_length}"

    print("\n✓ All circle geometry tests passed!")


def test_circle_orientation():
    """Test RMF orientation along circle."""
    path = create_horizontal_circle(radius=0.3, height=0.3, speed=0.2)

    print("\n" + "=" * 60)
    print("Testing Circle Orientation (RMF)")
    print("=" * 60)

    # Sample orientations around circle
    n_samples = 8
    for i in range(n_samples):
        s = i * path.total_length / n_samples
        quat = path.orientation(s)

        # Check quaternion is unit
        norm = np.linalg.norm(quat)
        print(f"s={s:.3f}: quat={quat}, norm={norm:.6f}")
        assert np.isclose(norm, 1.0, atol=1e-6), \
            f"Quaternion not unit: norm={norm}"

    print("\n✓ All orientation tests passed!")


def test_lookahead():
    """Test lookahead point generation."""
    path = create_horizontal_circle(radius=0.3, height=0.3, speed=0.2)

    print("\n" + "=" * 60)
    print("Testing Lookahead Points")
    print("=" * 60)

    s = 0.0
    n_points = 5
    ds = 0.02  # 2cm spacing

    lookahead = path.lookahead_points(s, n_points, ds)
    print(f"Starting at s={s}")
    print(f"Lookahead points (n={n_points}, ds={ds}m):")
    print(lookahead)

    assert lookahead.shape == (n_points, 3), \
        f"Wrong shape: {lookahead.shape}, expected ({n_points}, 3)"

    # Verify points are spaced correctly
    for i in range(n_points):
        expected_pos = path.position(s + (i+1) * ds)
        actual_pos = lookahead[i]
        diff = np.linalg.norm(expected_pos - actual_pos)
        print(f"  Point {i+1}: diff={diff:.6e}")
        assert diff < 1e-6, f"Lookahead point {i} incorrect"

    print("\n✓ Lookahead test passed!")


def test_trajectory_sampling():
    """Test sampling full trajectory."""
    path = create_horizontal_circle(radius=0.3, height=0.3, speed=0.2)

    print("\n" + "=" * 60)
    print("Testing Trajectory Sampling")
    print("=" * 60)

    duration = 2.0  # 2 seconds
    dt = 0.01  # 100 Hz

    traj = path.sample_trajectory(duration, dt)

    print(f"Duration: {duration}s, dt={dt}s")
    print(f"Positions shape: {traj['positions'].shape}")
    print(f"Velocities shape: {traj['velocities'].shape}")
    print(f"Orientations shape: {traj['orientations'].shape}")
    print(f"Arc lengths shape: {traj['arc_lengths'].shape}")

    n_steps = int(duration / dt)
    assert traj['positions'].shape == (n_steps, 3)
    assert traj['velocities'].shape == (n_steps, 3)
    assert traj['orientations'].shape == (n_steps, 4)
    assert traj['arc_lengths'].shape == (n_steps,)

    # Check that arc length increases
    s_values = traj['arc_lengths']
    print(f"\nArc length progression:")
    print(f"  Start: s={s_values[0]:.6f}")
    print(f"  End:   s={s_values[-1]:.6f}")
    print(f"  Expected distance traveled: {path.speed * duration:.6f}m")

    # Arc length should increase by speed * actual_duration
    # Note: n_steps samples give times from 0 to (n_steps-1)*dt
    actual_duration = (n_steps - 1) * dt
    expected_s_end = path.speed * actual_duration

    # Handle wrapping
    actual_distance = s_values[-1] - s_values[0]
    if actual_distance < 0:
        actual_distance += path.total_length

    print(f"  Actual distance: {actual_distance:.6f}m")
    print(f"  Actual time span: {actual_duration:.6f}s (0 to {(n_steps-1)*dt}s)")
    assert np.isclose(actual_distance, expected_s_end, atol=1e-4), \
        f"Incorrect distance traveled: {actual_distance:.6f} vs {expected_s_end:.6f}"

    print("\n✓ Trajectory sampling test passed!")


if __name__ == "__main__":
    try:
        test_circle_geometry()
        test_circle_orientation()
        test_lookahead()
        test_trajectory_sampling()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
