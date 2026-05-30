# Arm End-Effector Path Tracking

RL-based smooth end-effector tracking system with residual control and CAPS regularization.

## Overview

This project implements a reinforcement learning controller for tracking smooth Cartesian trajectories with a 7-DOF robotic arm (Franka Panda) in MuJoCo simulation. The core design prevents jitter structurally through:

- **Residual control in Cartesian twist space**: Policy outputs small corrections on top of feedforward motion
- **DLS Jacobian layer**: Smooth redundancy resolution with adaptive damping near singularities
- **CAPS regularization**: Temporal and spatial action smoothness in policy loss

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

## Headless Training

```bash
# Set MuJoCo to headless mode
export MUJOCO_GL=egl

# Run baseline circle tracking
python scripts/train_circle.py
```

## Progressive Build Order

1. **Step 1**: Circle baseline (position tracking only)
2. **Step 2**: Add CAPS regularization (measure jitter reduction)
3. **Step 3**: Velocity matching + figure-8 paths
4. **Step 4**: Full 6-DOF with orientation tracking
5. **Step 5**: Moving targets (generalization)

## Project Structure

```
src/
├── environment/    # Gymnasium environment
├── control/        # DLS Jacobian layer
├── paths/          # Path representation (circle, figure-8, etc.)
├── rl/             # PPO algorithm with CAPS
├── rewards/        # Tracking reward computation
└── utils/          # Config, kinematics, metrics

scripts/            # Training scripts for each build step
configs/            # YAML configuration files
tests/              # Unit tests
```

## Key Design Principles

- **Jitter is an action-space problem**: Prevent it structurally, not with reward penalties
- **ACTION_SCALE**: Primary smoothness lever (hard bounds EE deviation per step)
- **Velocity matching**: Move *with* the path, not just *near* it
- **EE-frame observations**: Generalize across workspace positions

## License

MIT
