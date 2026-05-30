# Quick Start Guide

## Step 1: Circle Baseline Training

You now have a complete implementation of the Step 1 baseline system!

### What's Implemented

✅ **Core Components**:
- DLS Jacobian controller with adaptive damping
- Circle path with arc-length parameterization & RMF orientation
- 58-dim observation builder (EE-frame for generalization)
- Position-only reward (bounded exponential)
- PPO algorithm with CAPS integration point
- Gymnasium environment with residual-feedforward control
- Franka Panda 7-DOF robot model

✅ **Progressive Build Order Ready**:
- Step 1: Circle baseline (current)
- Step 2: Add CAPS (set `caps.enabled: true` in config)
- Step 3: Add velocity matching (set `reward.w_vel > 0`)
- Step 4: Add orientation (set `reward.w_ori > 0`)
- Step 5: Moving targets (randomize path params)

### Training

```bash
# Make sure conda environment is activated
conda activate arm_tracking

# Set MuJoCo to headless mode (if not in script already)
export MUJOCO_GL=egl

# Run Step 1 training
python scripts/train_circle.py
```

### What to Expect

**Training**:
- 1M timesteps (~500 updates with n_steps=2048)
- Logs every 1k steps
- Evaluation every 10k steps
- Checkpoints every 50k steps

**Outputs** (in `outputs/circle_baseline/`):
- Checkpoints: `checkpoint_*.pt`
- Final model: `final_model.pt`
- Training logs (console output)

**Key Metrics** (from evaluation):
- Mean position error (should decrease to ~few mm)
- Integrated squared jerk (measures smoothness)
- High-freq power ratio (jitter indicator)

### Configuration

Edit `configs/circle_baseline.yaml` to adjust:
- **ACTION_SCALE**: Primary smoothness lever [0.02m linear, 0.10rad angular]
- **Path parameters**: radius, center, speed
- **Reward weights**: w_pos, w_action_rate, w_joint_vel
- **PPO hyperparameters**: learning_rate, n_steps, clip_epsilon, etc.

### Key Design Principles

1. **Action=0 is safe**: Yields pure feedforward tangent following
2. **EE-frame observations**: Generalizes across workspace positions
3. **DLS damping**: Smooth singularity handling (λ² increases as manipulability → 0)
4. **Bounded rewards**: Exponential form encourages settling, not infinite accumulation

### Progressive Build Order

**Step 2 (Add CAPS)**:
```yaml
# In configs/circle_baseline.yaml
caps:
  enabled: true
  lambda_temporal: 0.01
  lambda_spatial: 0.01
```
Expected: 60-80% reduction in high-freq power

**Step 3 (Velocity Matching)**:
```yaml
reward:
  w_vel: 1.0  # Enable velocity matching
```
Switch path to figure-8 for testing crossover smoothness

**Step 4 (Orientation)**:
```yaml
reward:
  w_ori: 1.0  # Enable orientation tracking
```
Full 6-DOF tracking with RMF

### Troubleshooting

**MuJoCo model not found**:
```bash
# Ensure git submodule is initialized
git submodule update --init --recursive
```

**Import errors**:
```bash
# Install in editable mode
pip install -e .
```

**GPU/CUDA**:
- Training uses CPU by default (PPO is not GPU-intensive for this problem size)
- If you have CUDA, PyTorch will auto-detect and use it

### Next Steps After Step 1

1. **Analyze jitter**: Compare actions before/after CAPS (Step 2)
2. **Implement figure-8 path**: For testing velocity matching (Step 3)
3. **Add orientation tracking**: Full 6-DOF control (Step 4)
4. **Generalize**: Randomize path parameters (Step 5)
5. **Visualization**: Create separate viz package to render trained policies

### File Structure

```
arm_path_tracking/
├── src/
│   ├── environment/     # ✓ Gymnasium env, observation, state
│   ├── control/         # ✓ DLS Jacobian layer
│   ├── paths/           # ✓ Circle (+ figure-8 for Step 3)
│   ├── rl/              # ✓ PPO, networks, CAPS
│   ├── rewards/         # ✓ Tracking reward
│   ├── utils/           # ✓ Kinematics, metrics, config
│   └── models/          # ✓ Franka Panda (mujoco_menagerie)
├── scripts/             # ✓ train_circle.py
├── configs/             # ✓ circle_baseline.yaml
├── tests/               # ✓ Unit tests (kinematics, DLS, circle)
└── outputs/             # Created during training

```

### Testing Before Training

Run unit tests to verify components:
```bash
pytest tests/ -v
# or
python tests/test_kinematics.py
python tests/test_circle_path.py
python tests/test_dls_controller.py
```

All tests should pass ✓

---

**You're ready to train!** 🚀

The system is designed to be:
- **Headless**: No GUI, runs on servers
- **Modular**: Each component independently tested
- **Progressive**: Build complexity step-by-step
- **Research-grade**: Implements CAPS, DLS, RMF from papers
