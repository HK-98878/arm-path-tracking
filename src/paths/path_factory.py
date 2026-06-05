"""Factory function for creating path instances from config."""

import numpy as np
from typing import Dict, List, Optional, Union

from .base_path import Path
from .circle_path import CirclePath
from .figure8_path import Figure8Path
from .bspline_path import BSplinePath


def create_path(
    path_type: str,
    center: np.ndarray,
    radius: float,
    speed: float,
    normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
    orientation_modes: Optional[List[str]] = None,
    rock_amplitude: float = 0.175,
    n_oscillations: int = 2,
    half_width: Optional[float] = None,
) -> Path:
    """Create a path instance from type string and parameters.

    Args:
        path_type: Type of path ('circle' or 'figure8')
        center: (3,) center position in world frame
        radius: Circle radius (also used as figure-8 half_width if not specified)
        speed: Path speed in m/s (negative for reverse direction)
        normal: (3,) plane normal vector (default: XY plane)
        orientation_modes: List of orientation modes ['fixed', 'rock_x', 'rock_y']
        rock_amplitude: Rocking amplitude in radians
        n_oscillations: Number of rock cycles per lap
        half_width: Figure-8 half-width (defaults to radius if not specified)

    Returns:
        Path instance

    Raises:
        ValueError: If path_type is unknown
    """
    if orientation_modes is None:
        orientation_modes = ['fixed']

    center = np.array(center, dtype=np.float64)
    normal = np.array(normal, dtype=np.float64)

    if path_type == 'circle':
        return CirclePath(
            radius=radius,
            center=center,
            speed=speed,
            normal=normal,
            orientation_modes=orientation_modes,
            rock_amplitude=rock_amplitude,
            n_oscillations=n_oscillations,
        )
    elif path_type == 'figure8':
        # Use half_width if specified, otherwise fall back to radius
        actual_half_width = half_width if half_width is not None else radius
        return Figure8Path(
            half_width=actual_half_width,
            center=center,
            speed=speed,
            normal=normal,
            orientation_modes=orientation_modes,
            rock_amplitude=rock_amplitude,
            n_oscillations=n_oscillations,
        )
    elif path_type == 'bspline':
        raise ValueError(
            "BSplinePath cannot be created via create_path() — use create_bspline_path() instead, "
            "which requires a random generator and bspline config dict."
        )
    else:
        raise ValueError(f"Unknown path type: {path_type}. Supported: 'circle', 'figure8', 'bspline'")


def create_path_from_config(
    path_type: str,
    config,
    speed_override: Optional[float] = None,
    orientation_modes_override: Optional[List[str]] = None,
) -> Path:
    """Create a path instance from config object.

    Args:
        path_type: Type of path ('circle' or 'figure8')
        config: Config object with path.* attributes
        speed_override: Override speed from config (for curriculum stages)
        orientation_modes_override: Override orientation modes (for curriculum)

    Returns:
        Path instance
    """
    speed = speed_override if speed_override is not None else config.path.speed
    orientation_modes = (
        orientation_modes_override
        if orientation_modes_override is not None
        else getattr(config.path, 'orientation_modes', ['fixed'])
    )

    # Get figure8-specific config if available
    half_width = None
    if hasattr(config.path, 'figure8') and hasattr(config.path.figure8, 'half_width'):
        half_width = config.path.figure8.half_width

    return create_path(
        path_type=path_type,
        center=np.array(config.path.center),
        radius=config.path.radius,
        speed=speed,
        normal=getattr(config.path, 'normal', np.array([0.0, 0.0, 1.0])),
        orientation_modes=orientation_modes,
        rock_amplitude=getattr(config.path, 'rock_amplitude', 0.175),
        n_oscillations=getattr(config.path, 'n_oscillations', 2),
        half_width=half_width,
    )


def create_bspline_path(
    center: np.ndarray,
    speed: float,
    bspline_config: Dict,
    rng: np.random.Generator,
    min_curvature_radius_override: Optional[float] = None,
) -> BSplinePath:
    """Create a random BSplinePath from a config dict.

    Args:
        center: (3,) workspace centre (same as config.path.center)
        speed: Path speed in m/s
        bspline_config: Dict with keys workspace_radius, n_control_points,
                        n_arc_samples, workspace_violation_prob, violation_magnitude,
                        and optionally min_curvature_radius
        rng: NumPy random generator for reproducible sampling
        min_curvature_radius_override: If set, overrides bspline_config value

    Returns:
        BSplinePath instance
    """
    min_r = min_curvature_radius_override or bspline_config.get('min_curvature_radius', 0.05)
    return BSplinePath.random(
        center=np.array(center, dtype=np.float64),
        workspace_radius=bspline_config.get('workspace_radius', 0.18),
        speed=speed,
        n_control_points=bspline_config.get('n_control_points', 7),
        rng=rng,
        min_curvature_radius=min_r,
        workspace_violation_prob=bspline_config.get('workspace_violation_prob', 0.0),
        violation_magnitude=bspline_config.get('violation_magnitude', 0.0),
        n_arc_samples=bspline_config.get('n_arc_samples', 1000),
    )


def sample_path_type(
    path_types: List[str],
    path_weights: Optional[List[float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Sample a path type from the pool with optional weights.

    Args:
        path_types: List of path type strings
        path_weights: Optional weights (must sum to 1, or will be normalized)
        rng: NumPy random generator (for reproducibility)

    Returns:
        Selected path type string
    """
    if rng is None:
        rng = np.random.default_rng()

    if path_weights is None:
        # Uniform sampling
        return rng.choice(path_types)
    else:
        # Weighted sampling
        weights = np.array(path_weights, dtype=np.float64)
        weights = weights / weights.sum()  # Normalize
        return rng.choice(path_types, p=weights)
