"""Debug script to check initial robot pose and circle location."""

import os
import sys
from pathlib import Path
import numpy as np
import mujoco

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.paths.circle_path import CirclePath
from src.utils.config import load_config

# Set headless
os.environ['MUJOCO_GL'] = 'egl'

# Load config
config = load_config(project_root / "configs" / "circle_baseline.yaml")

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(
    str(project_root / config.env.model_path)
)
data = mujoco.MjData(model)

# Get EE body ID
ee_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    config.env.ee_body_name
)

print("="*60)
print("Initial Pose Debug")
print("="*60)

# Create circle
path = CirclePath(
    radius=config.path.radius,
    center=np.array(config.path.center),
    speed=config.path.speed
)

print(f"\nCircle configuration:")
print(f"  Center: {path.center}")
print(f"  Radius: {path.radius}m")
print(f"  Speed: {path.speed}m/s")
print(f"  Total length: {path.total_length:.3f}m")

# Test different initial configurations
configs = {
    "Current (my guess)": np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785]),
    "Home position": np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]),
    "Zeros": np.zeros(7),
}

print(f"\nTesting initial configurations:")
print("="*60)

for name, q_init in configs.items():
    # Reset and set joints
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = q_init
    data.qvel[:7] = 0.0
    mujoco.mj_forward(model, data)

    # Get EE position
    ee_pos = data.xpos[ee_body_id].copy()

    # Get circle start position
    circle_start = path.position(0.0)

    # Compute errors at different points on circle
    errors = []
    for s in np.linspace(0, path.total_length, 20):
        target = path.position(s)
        error = np.linalg.norm(ee_pos - target)
        errors.append(error)

    min_error = min(errors)

    print(f"\n{name}:")
    print(f"  Joint config: {q_init}")
    print(f"  EE position: {ee_pos}")
    print(f"  Circle start (s=0): {circle_start}")
    print(f"  Error to start: {np.linalg.norm(ee_pos - circle_start)*1000:.1f}mm")
    print(f"  Min error to circle: {min_error*1000:.1f}mm")

print("\n" + "="*60)
print("Analysis:")
print("="*60)

# Check reachability
print(f"\nCircle bounds:")
print(f"  X range: [{path.center[0]-path.radius:.2f}, {path.center[0]+path.radius:.2f}]")
print(f"  Y range: [{path.center[1]-path.radius:.2f}, {path.center[1]+path.radius:.2f}]")
print(f"  Z: {path.center[2]:.2f}")

print(f"\nRecommendations:")
print(f"  - Initial error should be <100mm")
print(f"  - If all configs show >200mm error, circle may be unreachable")
print(f"  - Consider moving circle closer to robot base")
