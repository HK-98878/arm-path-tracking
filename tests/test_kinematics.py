"""Unit tests for kinematics utilities."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.kinematics import (
    quat_multiply,
    quat_inverse,
    quat_to_rotvec,
    rotvec_to_quat,
    quat_to_matrix,
    matrix_to_quat,
    geodesic_angle,
    rotation_error_rotvec,
    normalize_quaternion
)


def test_quaternion_identity():
    """Test identity quaternion."""
    print("=" * 60)
    print("Testing Quaternion Identity")
    print("=" * 60)

    # Identity quaternion [x, y, z, w] = [0, 0, 0, 1]
    q_identity = np.array([0, 0, 0, 1])

    # Identity should produce identity matrix
    R = quat_to_matrix(q_identity)
    print(f"Identity quaternion: {q_identity}")
    print(f"Rotation matrix:\n{R}")

    assert np.allclose(R, np.eye(3)), "Identity quaternion should give identity matrix"
    print("✓ Identity test passed\n")


def test_quaternion_inverse():
    """Test quaternion inverse."""
    print("=" * 60)
    print("Testing Quaternion Inverse")
    print("=" * 60)

    # Rotation by 90° around Z-axis
    angle = np.pi / 2
    axis = np.array([0, 0, 1])
    rotvec = angle * axis
    q = rotvec_to_quat(rotvec)

    print(f"Original rotation: {np.degrees(angle):.1f}° around Z")
    print(f"Quaternion: {q}")

    # Inverse
    q_inv = quat_inverse(q)
    print(f"Inverse quaternion: {q_inv}")

    # q * q_inv should be identity
    q_product = quat_multiply(q, q_inv)
    q_identity = np.array([0, 0, 0, 1])

    # Quaternion can be ±[0,0,0,1], both represent identity
    matches_identity = np.allclose(q_product, q_identity) or \
                      np.allclose(q_product, -q_identity)

    print(f"q * q_inv = {q_product}")
    assert matches_identity, "q * q_inv should be identity"
    print("✓ Inverse test passed\n")


def test_rotation_vector_conversion():
    """Test rotation vector <-> quaternion conversion."""
    print("=" * 60)
    print("Testing Rotation Vector Conversion")
    print("=" * 60)

    # 45° rotation around [1, 1, 0] (normalized)
    angle = np.pi / 4
    axis = np.array([1, 1, 0])
    axis = axis / np.linalg.norm(axis)
    rotvec_original = angle * axis

    print(f"Original rotation vector: {rotvec_original}")
    print(f"Angle: {np.degrees(angle):.1f}°, Axis: {axis}")

    # Convert to quaternion and back
    q = rotvec_to_quat(rotvec_original)
    rotvec_recovered = quat_to_rotvec(q)

    print(f"Quaternion: {q}")
    print(f"Recovered rotation vector: {rotvec_recovered}")

    # Should match (up to sign ambiguity for 180° rotations)
    assert np.allclose(rotvec_original, rotvec_recovered, atol=1e-6), \
        "Rotation vector not recovered correctly"
    print("✓ Rotation vector conversion test passed\n")


def test_geodesic_angle():
    """Test geodesic angle computation."""
    print("=" * 60)
    print("Testing Geodesic Angle")
    print("=" * 60)

    # Identity
    q1 = np.array([0, 0, 0, 1])
    q2 = np.array([0, 0, 0, 1])
    angle = geodesic_angle(q1, q2)
    print(f"Identity -> Identity: angle = {np.degrees(angle):.6f}°")
    assert np.isclose(angle, 0.0, atol=1e-6), "Identity angle should be 0"

    # 90° rotation around Z
    q1 = np.array([0, 0, 0, 1])  # Identity
    q2 = rotvec_to_quat(np.array([0, 0, np.pi/2]))  # 90° Z
    angle = geodesic_angle(q1, q2)
    print(f"Identity -> 90° Z: angle = {np.degrees(angle):.6f}° (expected 90°)")
    assert np.isclose(angle, np.pi/2, atol=1e-6), "Should be 90°"

    # 180° rotation
    q1 = np.array([0, 0, 0, 1])
    q2 = rotvec_to_quat(np.array([0, 0, np.pi]))  # 180° Z
    angle = geodesic_angle(q1, q2)
    print(f"Identity -> 180° Z: angle = {np.degrees(angle):.6f}° (expected 180°)")
    assert np.isclose(angle, np.pi, atol=1e-5), "Should be 180°"

    print("✓ Geodesic angle test passed\n")


def test_rotation_error():
    """Test rotation error as rotation vector."""
    print("=" * 60)
    print("Testing Rotation Error (Rotation Vector)")
    print("=" * 60)

    # Current: Identity
    # Target: 45° around Z
    q_current = np.array([0, 0, 0, 1])
    target_angle = np.pi / 4
    q_target = rotvec_to_quat(np.array([0, 0, target_angle]))

    error_rotvec = rotation_error_rotvec(q_current, q_target)

    print(f"Current: Identity")
    print(f"Target: {np.degrees(target_angle):.1f}° around Z")
    print(f"Error rotation vector: {error_rotvec}")
    print(f"Error magnitude: {np.linalg.norm(error_rotvec):.6f} rad " +
          f"({np.degrees(np.linalg.norm(error_rotvec)):.1f}°)")

    # Error magnitude should equal target angle
    assert np.isclose(np.linalg.norm(error_rotvec), target_angle, atol=1e-6), \
        "Error magnitude incorrect"

    # Error should point in Z direction
    expected_error = np.array([0, 0, target_angle])
    assert np.allclose(error_rotvec, expected_error, atol=1e-6), \
        "Error direction incorrect"

    print("✓ Rotation error test passed\n")


def test_quaternion_normalization():
    """Test quaternion normalization."""
    print("=" * 60)
    print("Testing Quaternion Normalization")
    print("=" * 60)

    # Unnormalized quaternion
    q_unnorm = np.array([1, 2, 3, 4], dtype=float)
    print(f"Unnormalized: {q_unnorm}, norm={np.linalg.norm(q_unnorm):.6f}")

    q_norm = normalize_quaternion(q_unnorm)
    print(f"Normalized: {q_norm}, norm={np.linalg.norm(q_norm):.6f}")

    assert np.isclose(np.linalg.norm(q_norm), 1.0, atol=1e-10), \
        "Normalized quaternion should have unit norm"

    print("✓ Normalization test passed\n")


def test_matrix_quaternion_consistency():
    """Test that matrix and quaternion representations are consistent."""
    print("=" * 60)
    print("Testing Matrix <-> Quaternion Consistency")
    print("=" * 60)

    # Create rotation: 60° around [1, 2, 3]
    angle = np.pi / 3
    axis = np.array([1, 2, 3], dtype=float)
    axis = axis / np.linalg.norm(axis)
    rotvec = angle * axis

    q_original = rotvec_to_quat(rotvec)
    R = quat_to_matrix(q_original)
    q_recovered = matrix_to_quat(R)

    print(f"Original quaternion: {q_original}")
    print(f"Rotation matrix:\n{R}")
    print(f"Recovered quaternion: {q_recovered}")

    # Quaternions can differ by sign (±q represent same rotation)
    matches = np.allclose(q_original, q_recovered, atol=1e-6) or \
              np.allclose(q_original, -q_recovered, atol=1e-6)

    assert matches, "Quaternion not recovered from matrix"

    # Check rotation matrix is orthogonal
    RTR = R.T @ R
    print(f"\nR^T @ R (should be identity):\n{RTR}")
    assert np.allclose(RTR, np.eye(3), atol=1e-10), "Rotation matrix not orthogonal"

    # Check determinant is +1
    det = np.linalg.det(R)
    print(f"det(R) = {det:.10f} (should be 1)")
    assert np.isclose(det, 1.0, atol=1e-10), "Rotation matrix determinant not 1"

    print("✓ Matrix-quaternion consistency test passed\n")


if __name__ == "__main__":
    try:
        test_quaternion_identity()
        test_quaternion_inverse()
        test_rotation_vector_conversion()
        test_geodesic_angle()
        test_rotation_error()
        test_quaternion_normalization()
        test_matrix_quaternion_consistency()

        print("=" * 60)
        print("ALL KINEMATICS TESTS PASSED! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
