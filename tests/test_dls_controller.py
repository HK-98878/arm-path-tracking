"""Unit tests for DLS controller logic."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.control.dls_jacobian import DLSController
from src.utils.manipulability import compute_manipulability


def create_random_jacobian(n_joints=7, seed=42):
    """Create a random full-rank Jacobian for testing."""
    np.random.seed(seed)
    J = np.random.randn(6, n_joints)
    return J


def create_singular_jacobian(n_joints=7):
    """Create a near-singular Jacobian (one small singular value)."""
    np.random.seed(123)
    J = np.random.randn(6, n_joints)

    # Make nearly singular by zeroing out one singular direction
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s[-1] = 1e-6  # Very small singular value
    J = U @ np.diag(s) @ Vt

    return J


def test_dls_controller_init():
    """Test DLS controller initialization."""
    print("=" * 60)
    print("Testing DLS Controller Initialization")
    print("=" * 60)

    n_joints = 7
    q_min = np.full(n_joints, -2.8973)
    q_max = np.full(n_joints, 2.8973)

    controller = DLSController(
        n_joints=n_joints,
        q_limits=(q_min, q_max),
        k_limit=0.5,
        lambda_min=1e-4,
        lambda_max=0.1,
        manip_threshold=0.05
    )

    print(f"Controller initialized:")
    print(f"  n_joints: {controller.n_joints}")
    print(f"  k_limit: {controller.k_limit}")
    print(f"  lambda_min: {controller.lambda_min}")
    print(f"  lambda_max: {controller.lambda_max}")
    print(f"  manip_threshold: {controller.manip_threshold}")

    assert controller.n_joints == n_joints
    print("✓ Initialization test passed\n")


def test_adaptive_damping():
    """Test adaptive damping based on manipulability."""
    print("=" * 60)
    print("Testing Adaptive Damping")
    print("=" * 60)

    n_joints = 7
    q_limits = (np.full(n_joints, -2.8973), np.full(n_joints, 2.8973))

    controller = DLSController(
        n_joints=n_joints,
        q_limits=q_limits,
        lambda_min=1e-4,
        lambda_max=0.1,
        manip_threshold=0.05
    )

    # Test 1: Well-conditioned Jacobian (high manipulability)
    J_good = create_random_jacobian(n_joints)
    manip_good = compute_manipulability(J_good)
    lambda_sq_good = controller.adaptive_damping(J_good)

    print(f"Well-conditioned Jacobian:")
    print(f"  Manipulability: {manip_good:.6f}")
    print(f"  Damping λ²: {lambda_sq_good:.6e}")
    print(f"  Should be ≈ lambda_min ({controller.lambda_min:.6e})")

    # High manipulability should give minimum damping
    if manip_good > controller.manip_threshold:
        assert np.isclose(lambda_sq_good, controller.lambda_min), \
            "High manipulability should give lambda_min"

    # Test 2: Singular Jacobian (low manipulability)
    J_singular = create_singular_jacobian(n_joints)
    manip_singular = compute_manipulability(J_singular)
    lambda_sq_singular = controller.adaptive_damping(J_singular)

    print(f"\nNear-singular Jacobian:")
    print(f"  Manipulability: {manip_singular:.6e}")
    print(f"  Damping λ²: {lambda_sq_singular:.6e}")
    print(f"  Should be > lambda_min")

    # Low manipulability should increase damping
    assert lambda_sq_singular > controller.lambda_min, \
        "Low manipulability should increase damping"

    print("✓ Adaptive damping test passed\n")


def test_dls_pseudoinverse():
    """Test DLS pseudoinverse computation."""
    print("=" * 60)
    print("Testing DLS Pseudoinverse")
    print("=" * 60)

    n_joints = 7
    q_limits = (np.full(n_joints, -2.8973), np.full(n_joints, 2.8973))
    controller = DLSController(n_joints=n_joints, q_limits=q_limits)

    J = create_random_jacobian(n_joints)
    lambda_sq = 1e-4

    J_pinv = controller.dls_pseudoinverse(J, lambda_sq)

    print(f"Jacobian shape: {J.shape}")
    print(f"Pseudoinverse shape: {J_pinv.shape}")

    assert J_pinv.shape == (n_joints, 6), \
        f"Wrong pseudoinverse shape: {J_pinv.shape}"

    # Test: J_pinv @ J should be close to identity in task space
    product = J @ J_pinv
    print(f"\nJ @ J_pinv shape: {product.shape}")
    print(f"Should be close to 6x6 identity (with damping effects)")

    # With small damping, should be close to identity
    error = np.linalg.norm(product - np.eye(6))
    print(f"||J @ J_pinv - I|| = {error:.6f}")

    # Error should be small (not exact due to damping)
    assert error < 0.1, "J @ J_pinv too far from identity"

    print("✓ DLS pseudoinverse test passed\n")


def test_null_space_projector():
    """Test null-space projector."""
    print("=" * 60)
    print("Testing Null-Space Projector")
    print("=" * 60)

    n_joints = 7
    q_limits = (np.full(n_joints, -2.8973), np.full(n_joints, 2.8973))
    controller = DLSController(n_joints=n_joints, q_limits=q_limits)

    J = create_random_jacobian(n_joints)
    lambda_sq = 1e-4
    J_pinv = controller.dls_pseudoinverse(J, lambda_sq)

    N = controller.null_space_projector(J, J_pinv)

    print(f"Null-space projector shape: {N.shape}")
    assert N.shape == (n_joints, n_joints), "Wrong null-space shape"

    # Properties of null-space projector:
    # 1. N @ N ≈ N (idempotent, approximately with damping)
    NN = N @ N
    error_idempotent = np.linalg.norm(NN - N)
    print(f"\n||N @ N - N|| = {error_idempotent:.6f} (should be small)")

    # 2. J @ N should be small (null-space motion doesn't affect task)
    JN = J @ N
    error_null = np.linalg.norm(JN)
    print(f"||J @ N|| = {error_null:.6f} (should be small)")

    # These should be small with damping
    assert error_null < 0.1, "J @ N should be small (null-space property)"

    print("✓ Null-space projector test passed\n")


def test_step_basic():
    """Test basic step computation."""
    print("=" * 60)
    print("Testing DLS Controller Step")
    print("=" * 60)

    n_joints = 7
    q_limits = (np.full(n_joints, -2.8973), np.full(n_joints, 2.8973))
    controller = DLSController(n_joints=n_joints, q_limits=q_limits)

    # Current configuration (middle of range)
    q_current = np.zeros(n_joints)

    # Desired EE twist (small motion)
    dx_ee = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])  # 1cm in X

    # Random Jacobian
    J = create_random_jacobian(n_joints)

    # Compute command
    q_cmd = controller.step(dx_ee, q_current, J, use_null_space=False)

    print(f"Current q: {q_current}")
    print(f"Desired dx_ee: {dx_ee}")
    print(f"Commanded q: {q_cmd}")
    print(f"Joint change: {q_cmd - q_current}")

    # Check output shape
    assert q_cmd.shape == (n_joints,), "Wrong command shape"

    # Check within limits
    assert np.all(q_cmd >= q_limits[0]) and np.all(q_cmd <= q_limits[1]), \
        "Command outside joint limits"

    # Check that command changed from current
    assert not np.allclose(q_cmd, q_current), \
        "Command should differ from current (non-zero desired motion)"

    print("✓ Basic step test passed\n")


def test_diagnostic_info():
    """Test diagnostic information."""
    print("=" * 60)
    print("Testing Diagnostic Info")
    print("=" * 60)

    n_joints = 7
    q_limits = (np.full(n_joints, -2.8973), np.full(n_joints, 2.8973))
    controller = DLSController(n_joints=n_joints, q_limits=q_limits)

    q_current = np.zeros(n_joints)
    J = create_random_jacobian(n_joints)

    info = controller.get_info(J, q_current)

    print("Diagnostic info:")
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    # Check keys exist
    required_keys = [
        'manipulability',
        'damping_lambda_sq',
        'condition_number',
        'smallest_singular_value',
        'joint_limit_proximity',
        'near_singularity'
    ]

    for key in required_keys:
        assert key in info, f"Missing diagnostic key: {key}"

    print("\n✓ Diagnostic info test passed\n")


if __name__ == "__main__":
    try:
        test_dls_controller_init()
        test_adaptive_damping()
        test_dls_pseudoinverse()
        test_null_space_projector()
        test_step_basic()
        test_diagnostic_info()

        print("=" * 60)
        print("ALL DLS CONTROLLER TESTS PASSED! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
