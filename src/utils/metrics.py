"""Metrics for evaluating trajectory smoothness and jitter."""

import numpy as np
from typing import Dict, Optional


def compute_jitter_metrics(
    action_sequence: np.ndarray,
    dt: float,
    freq_threshold: float = 5.0
) -> Dict[str, float]:
    """Compute jitter metrics from action time series (headless, no visualization).

    Args:
        action_sequence: (T, action_dim) array of actions over time
        dt: Time step (seconds)
        freq_threshold: Frequency (Hz) above which to measure high-freq power

    Returns:
        Dictionary with jitter metrics:
        - integrated_squared_jerk: ∫ ||jerk||^2 dt
        - max_action_derivative: max ||Δaction||
        - high_freq_power_ratio: Power above freq_threshold / total power
        - mean_action_change: mean ||Δaction||
    """
    if len(action_sequence) < 3:
        return {
            'integrated_squared_jerk': 0.0,
            'max_action_derivative': 0.0,
            'high_freq_power_ratio': 0.0,
            'mean_action_change': 0.0
        }

    # First derivative (action rate)
    action_dot = np.diff(action_sequence, axis=0) / dt  # (T-1, action_dim)

    # Second derivative (action acceleration = jerk in action space)
    action_ddot = np.diff(action_dot, axis=0) / dt  # (T-2, action_dim)

    # Integrated squared jerk
    jerk_squared = np.sum(action_ddot ** 2, axis=1)  # (T-2,)
    integrated_squared_jerk = np.trapezoid(jerk_squared, dx=dt)

    # Max and mean action derivative
    action_change_magnitude = np.linalg.norm(action_dot, axis=1)  # (T-1,)
    max_action_derivative = np.max(action_change_magnitude)
    mean_action_change = np.mean(action_change_magnitude)

    # FFT to measure high-frequency content
    # Compute for each action dimension, then average
    high_freq_ratios = []
    for dim in range(action_sequence.shape[1]):
        signal = action_sequence[:, dim]
        fft = np.fft.rfft(signal)
        power_spectrum = np.abs(fft) ** 2

        # Frequency bins
        freqs = np.fft.rfftfreq(len(signal), d=dt)

        # Total power
        total_power = np.sum(power_spectrum)

        if total_power > 0:
            # High-frequency power (above threshold)
            high_freq_mask = freqs > freq_threshold
            high_freq_power = np.sum(power_spectrum[high_freq_mask])

            high_freq_ratios.append(high_freq_power / total_power)
        else:
            high_freq_ratios.append(0.0)

    high_freq_power_ratio = np.mean(high_freq_ratios)

    return {
        'integrated_squared_jerk': float(integrated_squared_jerk),
        'max_action_derivative': float(max_action_derivative),
        'high_freq_power_ratio': float(high_freq_power_ratio),
        'mean_action_change': float(mean_action_change)
    }


def compute_tracking_error_metrics(
    positions: np.ndarray,
    targets: np.ndarray,
    orientations: Optional[np.ndarray] = None,
    target_orientations: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute tracking error metrics.

    Args:
        positions: (T, 3) actual EE positions
        targets: (T, 3) target positions
        orientations: Optional (T, 4) actual EE quaternions [x,y,z,w]
        target_orientations: Optional (T, 4) target quaternions

    Returns:
        Dictionary with tracking metrics:
        - mean_position_error: Mean L2 position error
        - max_position_error: Max L2 position error
        - rms_position_error: RMS position error
        - mean_orientation_error: Mean geodesic angle error (if provided)
        - max_orientation_error: Max geodesic angle error (if provided)
    """
    from .kinematics import geodesic_angle

    position_errors = np.linalg.norm(positions - targets, axis=1)  # (T,)

    metrics = {
        'mean_position_error': float(np.mean(position_errors)),
        'max_position_error': float(np.max(position_errors)),
        'rms_position_error': float(np.sqrt(np.mean(position_errors ** 2))),
    }

    # Orientation errors if provided
    if orientations is not None and target_orientations is not None:
        orientation_errors = []
        for q_actual, q_target in zip(orientations, target_orientations):
            angle = geodesic_angle(q_actual, q_target)
            orientation_errors.append(angle)

        orientation_errors = np.array(orientation_errors)
        metrics['mean_orientation_error'] = float(np.mean(orientation_errors))
        metrics['max_orientation_error'] = float(np.max(orientation_errors))

    return metrics


def compute_ee_jerk_metrics(
    ee_positions: np.ndarray,
    dt: float,
) -> Dict[str, float]:
    """Compute jerk metrics from EE position time series (true kinematic jerk).

    Uses the third derivative of position, which is the physically meaningful
    jerk measure for motion smoothness regardless of controller type.

    Args:
        ee_positions: (T, 3) end-effector positions over time
        dt: Time step (seconds)

    Returns:
        Dictionary with:
        - integrated_squared_jerk: ∫ ||jerk||^2 dt
        - rms_jerk: RMS jerk magnitude
        - max_jerk: Peak jerk magnitude
    """
    if len(ee_positions) < 4:
        return {
            'integrated_squared_jerk': 0.0,
            'rms_jerk': 0.0,
            'max_jerk': 0.0,
        }

    vel = np.diff(ee_positions, axis=0) / dt       # (T-1, 3)
    accel = np.diff(vel, axis=0) / dt              # (T-2, 3)
    jerk = np.diff(accel, axis=0) / dt             # (T-3, 3)

    jerk_sq = np.sum(jerk ** 2, axis=1)            # (T-3,)
    integrated_squared_jerk = np.trapezoid(jerk_sq, dx=dt)
    rms_jerk = float(np.sqrt(np.mean(jerk_sq)))
    max_jerk = float(np.sqrt(np.max(jerk_sq)))

    return {
        'integrated_squared_jerk': float(integrated_squared_jerk),
        'rms_jerk': rms_jerk,
        'max_jerk': max_jerk,
    }


def compute_velocity_matching_error(
    ee_velocities: np.ndarray,
    target_velocities: np.ndarray
) -> Dict[str, float]:
    """Compute velocity matching error metrics.

    Args:
        ee_velocities: (T, 3) actual EE linear velocities
        target_velocities: (T, 3) target velocities

    Returns:
        Dictionary with velocity metrics:
        - mean_velocity_error: Mean L2 velocity error
        - rms_velocity_error: RMS velocity error
    """
    velocity_errors = np.linalg.norm(ee_velocities - target_velocities, axis=1)

    return {
        'mean_velocity_error': float(np.mean(velocity_errors)),
        'rms_velocity_error': float(np.sqrt(np.mean(velocity_errors ** 2))),
    }


def compare_jitter_reduction(
    baseline_actions: np.ndarray,
    caps_actions: np.ndarray,
    dt: float
) -> Dict[str, float]:
    """Compare jitter between baseline and CAPS runs.

    Args:
        baseline_actions: (T, action_dim) actions without CAPS
        caps_actions: (T, action_dim) actions with CAPS
        dt: Time step

    Returns:
        Dictionary with reduction percentages:
        - jerk_reduction_pct: % reduction in integrated squared jerk
        - high_freq_reduction_pct: % reduction in high-frequency power
    """
    baseline_metrics = compute_jitter_metrics(baseline_actions, dt)
    caps_metrics = compute_jitter_metrics(caps_actions, dt)

    jerk_baseline = baseline_metrics['integrated_squared_jerk']
    jerk_caps = caps_metrics['integrated_squared_jerk']

    hf_baseline = baseline_metrics['high_freq_power_ratio']
    hf_caps = caps_metrics['high_freq_power_ratio']

    jerk_reduction = 0.0 if jerk_baseline == 0 else \
        100 * (1 - jerk_caps / jerk_baseline)

    hf_reduction = 0.0 if hf_baseline == 0 else \
        100 * (1 - hf_caps / hf_baseline)

    return {
        'jerk_reduction_pct': float(jerk_reduction),
        'high_freq_reduction_pct': float(hf_reduction),
        'baseline_jerk': float(jerk_baseline),
        'caps_jerk': float(jerk_caps),
        'baseline_high_freq': float(hf_baseline),
        'caps_high_freq': float(hf_caps)
    }
