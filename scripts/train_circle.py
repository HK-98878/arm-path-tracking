"""Training script for Step 1: Circle baseline (position tracking only).

This script implements the first step of the progressive build order:
- Circle path tracking
- Position-only reward
- Standard PPO (no CAPS)
- DLS controller
- Headless training
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.ee_tracking_env import EETrackingEnv
from src.paths import create_path, sample_path_type, create_bspline_path
from src.rl.ppo import PPO
from src.utils.config import load_config
from src.utils.metrics import compute_jitter_metrics, compute_tracking_error_metrics, compute_ee_jerk_metrics
from src.utils.normalization import RunningMeanStd, RewardNormalizer


class CurriculumManager:
    """Manages curriculum learning stage transitions."""

    def __init__(self, config):
        self.enabled = getattr(config, 'curriculum', None) is not None and getattr(config.curriculum, 'enabled', False)
        if not self.enabled:
            self.current_stage = 0
            return

        self.stages = list(config.curriculum.stages)
        self.consecutive_evals = config.curriculum.consecutive_evals
        self.warmup_steps = config.curriculum.warmup_after_transition
        self.current_stage = 0
        self.steps_in_stage = 0
        self.eval_history = []  # List of dicts: {path_type: pos_error_mean}

    def get_current_params(self):
        """Get all parameters for current stage.

        Returns:
            dict with speed, sig_pos, and optionally sig_ori, w_ori
            or empty dict if curriculum disabled
        """
        if not self.enabled:
            return {}
        return dict(self.stages[self.current_stage])

    def record_eval(self, per_path_errors: dict):
        """Record evaluation result. per_path_errors: {path_type: mean_pos_error}"""
        if self.enabled:
            self.eval_history.append(per_path_errors)

    def should_advance(self):
        """Check if should advance to next stage."""
        if not self.enabled or self.current_stage >= len(self.stages) - 1:
            return False

        stage = self.stages[self.current_stage]
        threshold = stage.get('advance_threshold')
        max_steps = stage.get('max_steps')  # None means no step-count fallback
        min_steps = stage.get('min_steps_in_stage', 0)

        # Never advance before the minimum step floor regardless of performance.
        if self.steps_in_stage < min_steps:
            return False

        # All active path types must independently meet the threshold for
        # consecutive_evals in a row.
        if threshold and len(self.eval_history) >= self.consecutive_evals:
            recent = self.eval_history[-self.consecutive_evals:]
            if all(
                all(err < threshold for err in eval_entry.values())
                for eval_entry in recent
            ):
                return True

        # Fallback: max steps in stage (skipped when max_steps is null/None)
        if max_steps is not None and self.steps_in_stage >= max_steps:
            return True

        return False

    def advance(self):
        """Advance to next stage. Returns new (speed, sig_pos)."""
        self.current_stage += 1
        self.steps_in_stage = 0
        self.eval_history = []
        return self.get_current_params()

    def step(self, n=1):
        """Increment step counter."""
        if self.enabled:
            self.steps_in_stage += n


class HyperparamScheduler:
    """Manages LR decay and entropy annealing with curriculum-aware warmup."""

    def __init__(self, config):
        """Initialize scheduler.

        Args:
            config: Configuration object with ppo and curriculum settings
        """
        # LR settings
        self.lr_initial = config.ppo.learning_rate
        self.lr_min = getattr(config.ppo, 'lr_min', self.lr_initial * 0.1)
        self.lr_decay = getattr(config.ppo, 'lr_decay', 'linear')

        # Entropy settings
        self.ent_initial = config.ppo.ent_coef
        self.ent_min = getattr(config.ppo, 'ent_coef_min', self.ent_initial * 0.1)
        self.ent_decay = getattr(config.ppo, 'ent_decay', 'linear')

        # Training duration
        self.total_timesteps = config.training.total_timesteps

        # Curriculum warmup settings (if curriculum enabled)
        self.curriculum_enabled = getattr(config, 'curriculum', None) is not None and getattr(config.curriculum, 'enabled', False)
        if self.curriculum_enabled:
            self.lr_warmup_mult = getattr(config.curriculum, 'lr_warmup_multiplier', 1.0)
            self.ent_warmup_mult = getattr(config.curriculum, 'ent_warmup_multiplier', 1.0)
            self.warmup_steps = config.curriculum.warmup_after_transition
        else:
            self.lr_warmup_mult = 1.0
            self.ent_warmup_mult = 1.0
            self.warmup_steps = 0

        # State for curriculum warmup
        self.steps_since_transition = float('inf')  # Start with no warmup active

    def on_curriculum_transition(self):
        """Called when curriculum advances to a new stage."""
        self.steps_since_transition = 0

    def step(self):
        """Increment warmup counter."""
        if self.steps_since_transition < float('inf'):
            self.steps_since_transition += 1

    def get_lr(self, timestep: int) -> float:
        """Get learning rate for given timestep.

        Args:
            timestep: Current training timestep

        Returns:
            Learning rate (with decay and warmup applied)
        """
        # Base LR with decay
        progress = timestep / self.total_timesteps
        if self.lr_decay == 'linear':
            base_lr = self.lr_initial + (self.lr_min - self.lr_initial) * progress
        else:
            base_lr = self.lr_initial  # No decay

        base_lr = max(base_lr, self.lr_min)

        # Apply warmup multiplier if in warmup period
        if self.steps_since_transition < self.warmup_steps:
            warmup_progress = self.steps_since_transition / self.warmup_steps
            # Linearly decay warmup multiplier from lr_warmup_mult to 1.0
            multiplier = self.lr_warmup_mult + (1.0 - self.lr_warmup_mult) * warmup_progress
            return base_lr * multiplier

        return base_lr

    def get_entropy_coef(self, timestep: int) -> float:
        """Get entropy coefficient for given timestep.

        Args:
            timestep: Current training timestep

        Returns:
            Entropy coefficient (with annealing and warmup applied)
        """
        # Base entropy with annealing
        progress = timestep / self.total_timesteps
        if self.ent_decay == 'linear':
            base_ent = self.ent_initial + (self.ent_min - self.ent_initial) * progress
        else:
            base_ent = self.ent_initial  # No decay

        base_ent = max(base_ent, self.ent_min)

        # Apply warmup multiplier if in warmup period
        if self.steps_since_transition < self.warmup_steps:
            warmup_progress = self.steps_since_transition / self.warmup_steps
            # Linearly decay warmup multiplier from ent_warmup_mult to 1.0
            multiplier = self.ent_warmup_mult + (1.0 - self.ent_warmup_mult) * warmup_progress
            return base_ent * multiplier

        return base_ent


def check_for_spike(current_error, previous_error, threshold_ratio=2.0):
    """Detect if error has spiked dramatically."""
    if previous_error is not None and previous_error > 0:
        if current_error > previous_error * threshold_ratio:
            return True
    return False


def make_env(config, bidirectional=False, include_orientation=False, path_type='circle'):
    """Create environment from config.

    Args:
        config: Configuration object
        bidirectional: If True, randomly negate speed for this env
        include_orientation: Whether to enable orientation control
        path_type: Type of path to create ('circle' or 'figure8')

    Returns:
        EETrackingEnv instance
    """
    # Randomize direction if bidirectional
    speed = config.path.speed
    if bidirectional and np.random.random() < 0.5:
        speed = -speed

    # Get orientation variation parameters
    orientation_modes = getattr(config.path, 'orientation_modes', ['fixed'])
    rock_amplitude = getattr(config.path, 'rock_amplitude', 0.175)
    n_oscillations = getattr(config.path, 'n_oscillations', 2)

    # Create path using factory
    path = create_path(
        path_type=path_type,
        center=np.array(config.path.center),
        radius=config.path.radius,
        speed=speed,
        orientation_modes=orientation_modes,
        rock_amplitude=rock_amplitude,
        n_oscillations=n_oscillations,
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
        render_mode=None,  # Headless
        dls_config=config.control.dls.to_dict() if hasattr(config.control, 'dls') else None,
        include_orientation=include_orientation,
        lookahead_n=getattr(config.env, 'lookahead_n', 5),
        lookahead_ds=getattr(config.env, 'lookahead_ds', 0.02),
        randomize_start_position=getattr(config.env, 'randomize_start_position', False),
        start_position_noise=getattr(config.env, 'start_position_noise', 0.06),
    )

    return env


def make_env_with_stage(config, stage_params, bidirectional=False, path_type_override=None):
    """Create environment with curriculum stage parameters.

    Args:
        config: Configuration object
        stage_params: dict with speed, sig_pos, and optionally sig_ori, w_ori,
                      orientation_modes, path_types, path_weights
        bidirectional: If True, randomly negate speed
        path_type_override: If provided, use this path type instead of sampling

    Returns:
        EETrackingEnv instance, selected path_type
    """
    # Get speed from stage params
    speed = stage_params.get('speed', config.path.speed)

    # Randomize direction if bidirectional
    if bidirectional and np.random.random() < 0.5:
        speed = -speed

    # Get orientation variation parameters (stage overrides config defaults)
    orientation_modes = stage_params.get(
        'orientation_modes',
        getattr(config.path, 'orientation_modes', ['fixed'])
    )
    rock_amplitude = getattr(config.path, 'rock_amplitude', 0.175)
    n_oscillations = getattr(config.path, 'n_oscillations', 2)

    # Select path type from pool (or use override)
    if path_type_override is not None:
        path_type = path_type_override
    else:
        path_types = stage_params.get('path_types', ['circle'])
        path_weights = stage_params.get('path_weights', None)
        path_type = sample_path_type(path_types, path_weights)

    # Create path using factory
    bspline_cfg = None
    if path_type == 'bspline':
        bspline_cfg = getattr(config, 'bspline_path', None)
        bspline_cfg = bspline_cfg.to_dict() if bspline_cfg is not None else {}
        min_r = stage_params.get('min_curvature_radius', bspline_cfg.get('min_curvature_radius', 0.05))
        path = create_bspline_path(
            center=np.array(config.path.center),
            speed=speed,
            bspline_config=bspline_cfg,
            rng=np.random.default_rng(0),  # initial path; env regenerates on reset
            min_curvature_radius_override=min_r,
            orientation_modes=orientation_modes,
        )
    else:
        path = create_path(
            path_type=path_type,
            center=np.array(config.path.center),
            radius=config.path.radius,
            speed=speed,
            orientation_modes=orientation_modes,
            rock_amplitude=rock_amplitude,
            n_oscillations=n_oscillations,
        )

    # Build reward config with stage overrides
    reward_config = config.reward.to_dict()
    if stage_params.get('sig_pos') is not None:
        reward_config['sig_pos'] = stage_params['sig_pos']
    if stage_params.get('sig_ori') is not None:
        reward_config['sig_ori'] = stage_params['sig_ori']
    if stage_params.get('w_ori') is not None:
        reward_config['w_ori'] = stage_params['w_ori']

    # Determine if orientation control is enabled
    include_orientation = reward_config.get('w_ori', 0) > 0

    # Build per-stage observation noise config
    obs_noise_config = {
        'obs_noise_pos_std': stage_params.get('obs_noise_pos_std', 0.0),
        'obs_noise_vel_std': stage_params.get('obs_noise_vel_std', 0.0),
        'lookahead_noise_std': stage_params.get('lookahead_noise_std', 0.0),
    }
    has_noise = any(v > 0 for v in obs_noise_config.values())

    # Whether to include EE acceleration in observation
    include_ee_accel = getattr(getattr(config, 'env', None), 'include_ee_accel', False)

    # Create environment
    env = EETrackingEnv(
        model_path=str(project_root / config.env.model_path),
        path=path,
        reward_config=reward_config,
        action_scale=np.array(config.control.action_scale),
        dt=config.env.dt,
        max_episode_steps=config.env.max_episode_steps,
        ee_body_name=config.env.ee_body_name,
        render_mode=None,  # Headless
        dls_config=config.control.dls.to_dict() if hasattr(config.control, 'dls') else None,
        include_orientation=include_orientation,
        lookahead_n=getattr(config.env, 'lookahead_n', 5),
        lookahead_ds=getattr(config.env, 'lookahead_ds', 0.02),
        randomize_start_position=getattr(config.env, 'randomize_start_position', False),
        start_position_noise=getattr(config.env, 'start_position_noise', 0.06),
        include_ee_accel=include_ee_accel,
        obs_noise_config=obs_noise_config if has_noise else None,
    )

    # Store bspline config on env for per-episode regeneration in reset()
    if path_type == 'bspline' and bspline_cfg is not None:
        env._bspline_config = {
            **bspline_cfg,
            'center': np.array(config.path.center),
            'speed': speed,
            'min_curvature_radius': stage_params.get(
                'min_curvature_radius', bspline_cfg.get('min_curvature_radius', 0.05)
            ),
            'orientation_modes': orientation_modes,
        }

    return env, path_type


def evaluate(env, agent, obs_rms, num_episodes=5, eval_seed=None):
    """Evaluate agent performance.

    Args:
        env: Environment
        agent: PPO agent
        obs_rms: Observation normalization statistics
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    all_actions = []
    all_positions = []
    all_targets = []

    # Reward component accumulators
    all_r_pos = []
    all_r_ori = []
    all_r_vel = []
    all_p_action_rate = []
    all_p_joint_vel = []
    all_pos_errors = []
    all_ori_errors = []

    for i in range(num_episodes):
        obs, _ = env.reset(seed=eval_seed if i == 0 else None)
        obs = obs_rms.normalize(obs)  # Normalize observation
        done = False
        episode_reward = 0
        episode_length = 0
        episode_actions = []
        episode_positions = []
        episode_targets = []

        # Per-episode reward components
        ep_r_pos = []
        ep_r_ori = []
        ep_r_vel = []
        ep_p_action_rate = []
        ep_p_joint_vel = []
        ep_pos_errors = []
        ep_ori_errors = []

        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)

            # Store for metrics
            episode_actions.append(action)

            # Get current state for tracking
            state = env._get_robot_state()
            target_pos = env.path.position(env.s_current)
            episode_positions.append(state.ee_pos_world.copy())
            episode_targets.append(target_pos.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            obs = obs_rms.normalize(obs)  # Normalize observation
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Collect reward components from info
            ep_r_pos.append(info.get('r_pos', 0))
            ep_r_ori.append(info.get('r_ori', 0))
            ep_r_vel.append(info.get('r_vel', 0))
            ep_p_action_rate.append(info.get('p_action_rate', 0))
            ep_p_joint_vel.append(info.get('p_joint_vel', 0))
            ep_pos_errors.append(info.get('pos_error', 0))
            ep_ori_errors.append(info.get('ori_error', 0))

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_actions.append(np.array(episode_actions))
        all_positions.append(np.array(episode_positions))
        all_targets.append(np.array(episode_targets))

        all_r_pos.append(np.mean(ep_r_pos))
        all_r_ori.append(np.mean(ep_r_ori))
        all_r_vel.append(np.mean(ep_r_vel))
        all_p_action_rate.append(np.mean(ep_p_action_rate))
        all_p_joint_vel.append(np.mean(ep_p_joint_vel))
        all_pos_errors.extend(ep_pos_errors)
        all_ori_errors.extend(ep_ori_errors)

    # Aggregate metrics
    metrics = {
        'mean_episode_reward': float(np.mean(episode_rewards)),
        'std_episode_reward': float(np.std(episode_rewards)),
        'mean_episode_length': float(np.mean(episode_lengths)),
        # Reward components (averaged across episodes)
        'mean_r_pos': float(np.mean(all_r_pos)),
        'mean_r_ori': float(np.mean(all_r_ori)),
        'mean_r_vel': float(np.mean(all_r_vel)),
        'mean_p_action_rate': float(np.mean(all_p_action_rate)),
        'mean_p_joint_vel': float(np.mean(all_p_joint_vel)),
        # Position error statistics
        'pos_error_mean': float(np.mean(all_pos_errors)),
        'pos_error_std': float(np.std(all_pos_errors)),
        'pos_error_max': float(np.max(all_pos_errors)),
        'pos_error_p90': float(np.percentile(all_pos_errors, 90)),
        # Orientation error statistics (radians)
        'ori_error_mean': float(np.mean(all_ori_errors)),
        'ori_error_std': float(np.std(all_ori_errors)),
        'ori_error_max': float(np.max(all_ori_errors)) if all_ori_errors else 0.0,
    }

    # Jitter metrics (from first episode)
    if len(all_actions) > 0:
        jitter = compute_jitter_metrics(all_actions[0], env.dt)
        metrics.update({f'jitter_{k}': float(v) for k, v in jitter.items()})

    # EE kinematic jerk (physical, from positions — averaged across all episodes)
    if len(all_positions) > 0:
        ee_jerk_rms_values = []
        for positions in all_positions:
            ej = compute_ee_jerk_metrics(np.array(positions), env.dt)
            ee_jerk_rms_values.append(ej['rms_jerk'])
        metrics['ee_jerk_rms'] = float(np.mean(ee_jerk_rms_values))

    # Tracking error metrics (from first episode)
    if len(all_positions) > 0 and len(all_targets) > 0:
        tracking = compute_tracking_error_metrics(
            all_positions[0],
            all_targets[0]
        )
        metrics.update({f'tracking_{k}': float(v) for k, v in tracking.items()})

    return metrics


def evaluate_multi_path(config, stage_params, agent, obs_rms, num_episodes_per_path=3):
    """Evaluate agent on each path type in the current stage's pool.

    Args:
        config: Configuration object
        stage_params: Current curriculum stage parameters
        agent: PPO agent
        obs_rms: Observation normalization statistics
        num_episodes_per_path: Episodes to run per path type

    Returns:
        Dictionary with aggregate metrics and per-path breakdown
    """
    path_types = stage_params.get('path_types', ['circle'])

    per_path_metrics = {}
    all_pos_errors = []
    all_ori_errors = []

    for path_type in path_types:
        # Create env with specific path type
        env, _ = make_env_with_stage(
            config, stage_params, bidirectional=False,
            path_type_override=path_type
        )

        # B-splines are random per episode: run more to get a reliable average
        n_eps = max(5, num_episodes_per_path) if path_type == 'bspline' else num_episodes_per_path

        # Fixed seed for bspline eval so the same path sequence is used every eval,
        # making the trend signal interpretable (training paths remain random).
        bspline_eval_seed = config.seed if path_type == 'bspline' else None

        # Run evaluation
        metrics = evaluate(env, agent, obs_rms, n_eps, eval_seed=bspline_eval_seed)
        per_path_metrics[path_type] = metrics
        env.close()

        # Collect for aggregation
        all_pos_errors.append(metrics['pos_error_mean'])
        all_ori_errors.append(metrics['ori_error_mean'])

    # Aggregate metrics (average across path types)
    aggregate = {
        'pos_error_mean': float(np.mean(all_pos_errors)),
        'ori_error_mean': float(np.mean(all_ori_errors)),
        'per_path': per_path_metrics,
    }

    # Copy first path type's other metrics for compatibility
    first_path = path_types[0]
    for key in per_path_metrics[first_path]:
        if key not in aggregate:
            aggregate[key] = per_path_metrics[first_path][key]

    return aggregate


def train(config):
    """Main training loop.

    Args:
        config: Configuration object
    """
    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Set MuJoCo to headless mode
    os.environ['MUJOCO_GL'] = 'egl'

    # Create output directory
    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize training log
    training_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'reward': config.reward.to_dict(),
            'ppo': config.ppo.to_dict(),
            'path': config.path.to_dict(),
            'training': config.training.to_dict(),
            'caps': config.caps.to_dict() if hasattr(config, 'caps') else {'enabled': False},
        },
        'evaluations': [],
        'training_updates': [],
    }
    log_path = output_dir / 'training_log.json'

    def save_log():
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

    # Read bidirectional and orientation settings
    bidirectional = getattr(config.training, 'bidirectional', False)
    include_orientation = getattr(config.reward, 'w_ori', 0) > 0

    # Create environment
    print("Creating environment...")
    env = make_env(config, bidirectional=bidirectional, include_orientation=include_orientation)

    # Initialize curriculum manager
    curriculum = CurriculumManager(config)
    if curriculum.enabled:
        stage_params = curriculum.get_current_params()
        env.close()
        env, current_path_type = make_env_with_stage(config, stage_params, bidirectional=bidirectional)
        print(f"  Curriculum enabled - Stage 0: {stage_params}")
        print(f"  Initial path type: {current_path_type}")
    else:
        print("  Curriculum disabled - using config path speed")

    print(f"  Bidirectional: {bidirectional}")
    print(f"  Orientation control: {include_orientation}")

    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")

    # Create agent
    print("\nCreating PPO agent...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    agent = PPO(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        learning_rate=config.ppo.learning_rate,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_epsilon=config.ppo.clip_epsilon,
        n_steps=config.ppo.n_steps,
        n_epochs=config.ppo.n_epochs,
        batch_size=config.ppo.batch_size,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        hidden_sizes=tuple(config.ppo.hidden_sizes),
        device=device,
        caps_config=config.caps.to_dict() if hasattr(config, 'caps') else None
    )

    # Create hyperparameter scheduler for LR decay and entropy annealing
    scheduler = HyperparamScheduler(config)
    print(f"\nHyperparameter scheduling:")
    print(f"  LR: {scheduler.lr_initial:.2e} → {scheduler.lr_min:.2e} ({scheduler.lr_decay})")
    print(f"  Entropy: {scheduler.ent_initial:.3f} → {scheduler.ent_min:.3f} ({scheduler.ent_decay})")
    if curriculum.enabled:
        print(f"  Curriculum warmup: LR {scheduler.lr_warmup_mult}x, Ent {scheduler.ent_warmup_mult}x for {scheduler.warmup_steps} steps")

    print(f"\nTraining for {config.training.total_timesteps:,} timesteps...")
    print(f"  n_steps per update: {config.ppo.n_steps}")
    print(f"  Total updates: {config.training.total_timesteps // config.ppo.n_steps}")

    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    reward_normalizer = RewardNormalizer(gamma=config.ppo.gamma)

    # Warmup period for normalization (freeze after this)
    warmup_steps = getattr(config.training, 'normalization_warmup', 10000)
    print(f"  Using running observation normalization (freeze after {warmup_steps} steps)")
    print(f"  Using reward normalization")

    # Training loop
    obs, _ = env.reset()
    obs_rms.update(obs)
    obs = obs_rms.normalize(obs)
    episode_reward = 0
    episode_length = 0
    num_episodes = 0
    last_update_metrics = None

    # Spike detection tracking
    prev_eval_error = None
    spike_suppression_evals = 0  # Suppress spike detection for N evals after curriculum shift

    # Curriculum warmup tracking (next timestep when normalization should freeze)
    warmup_end_step = warmup_steps if not curriculum.enabled else warmup_steps

    for timestep in range(config.training.total_timesteps):
        # Curriculum and scheduler step counters
        curriculum.step()
        scheduler.step()

        # Update LR and entropy coefficient based on schedule
        current_lr = scheduler.get_lr(timestep)
        current_ent = scheduler.get_entropy_coef(timestep)
        agent.set_learning_rate(current_lr)
        agent.set_entropy_coef(current_ent)

        # Freeze normalization after warmup
        if timestep == warmup_end_step and not obs_rms.frozen:
            obs_rms.freeze()
            reward_normalizer.freeze()
            print(f"\n  [Step {timestep}] Normalization frozen.")

        # Select action
        action, value, log_prob = agent.select_action(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Normalize reward
        reward_normalized = reward_normalizer.normalize(reward, done)

        obs_rms.update(next_obs)
        next_obs_normalized = obs_rms.normalize(next_obs)

        # Store transition with normalized reward
        agent.store_transition(obs, action, reward_normalized, value, log_prob, done)

        episode_reward += reward
        episode_length += 1

        # Update observation
        obs = next_obs_normalized

        # Episode end
        if done:
            num_episodes += 1
            obs, _ = env.reset()
            obs_rms.update(obs)
            obs = obs_rms.normalize(obs)
            episode_reward = 0
            episode_length = 0

        # Update policy
        if (timestep + 1) % config.ppo.n_steps == 0:
            update_metrics = agent.update(obs, done)
            last_update_metrics = update_metrics  # Store for logging at eval time

            # Console log (sparse)
            if (timestep + 1) % config.training.log_frequency == 0:
                print(f"\nTimestep {timestep + 1:,} / {config.training.total_timesteps:,}")
                print(f"  Episodes: {num_episodes}")
                print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
                print(f"  Value loss: {update_metrics['value_loss']:.4f}")
                print(f"  Entropy: {-update_metrics['entropy_loss']:.4f}")
                print(f"  Approx KL: {update_metrics['approx_kl']:.4f}")
                print(f"  Clip fraction: {update_metrics['clip_fraction']:.4f}")
                print(f"  LR: {current_lr:.2e}, Ent coef: {current_ent:.4f}")

        # Evaluation
        if (timestep + 1) % config.training.eval_frequency == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at timestep {timestep + 1:,}")
            print('='*60)

            # Get current stage params for multi-path evaluation
            stage_params = curriculum.get_current_params() if curriculum.enabled else {}
            path_types = stage_params.get('path_types', ['circle'])

            if len(path_types) > 1:
                # Multi-path evaluation
                eval_metrics = evaluate_multi_path(
                    config, stage_params, agent, obs_rms,
                    num_episodes_per_path=max(2, config.training.num_eval_episodes // len(path_types))
                )
                # Print per-path breakdown (pos + EE RMS jerk)
                print(f"  Per-path metrics:")
                for pt, pm in eval_metrics.get('per_path', {}).items():
                    jerk_str = f"  jerk={pm['ee_jerk_rms']:.1f}" if 'ee_jerk_rms' in pm else ''
                    print(f"    {pt}: pos={pm['pos_error_mean']*1000:.2f}mm  ori={np.degrees(pm['ori_error_mean']):.2f}°{jerk_str}")
            else:
                # Single-path evaluation (original behavior)
                eval_metrics = evaluate(env, agent, obs_rms, config.training.num_eval_episodes)
                print(f"  Mean position error: {eval_metrics['tracking_mean_position_error']*1000:.2f} mm")
                print(f"  Max position error: {eval_metrics['tracking_max_position_error']*1000:.2f} mm")
                if eval_metrics.get('ori_error_mean', 0) > 0:
                    print(f"  Mean orientation error: {np.degrees(eval_metrics['ori_error_mean']):.2f} deg")

            print(f"  Mean episode reward: {eval_metrics['mean_episode_reward']:.2f} ± {eval_metrics['std_episode_reward']:.2f}")
            print(f"  Mean episode length: {eval_metrics['mean_episode_length']:.1f}")
            if 'ee_jerk_rms' in eval_metrics:
                print(f"  EE RMS jerk: {eval_metrics['ee_jerk_rms']:.1f} m/s³")
            print(f"  Reward components: r_pos={eval_metrics['mean_r_pos']:.3f}, r_ori={eval_metrics['mean_r_ori']:.3f}, r_vel={eval_metrics['mean_r_vel']:.3f}")
            print(f"  Penalties: action_rate={eval_metrics['mean_p_action_rate']:.2e}, joint_vel={eval_metrics['mean_p_joint_vel']:.2e}")

            # Log to JSON
            eval_entry = {'timestep': timestep + 1, **eval_metrics}
            training_log['evaluations'].append(eval_entry)

            # Log training metrics at eval time (every 10k instead of 256k)
            if last_update_metrics is not None:
                training_log['training_updates'].append({
                    'timestep': timestep + 1,
                    'episodes': num_episodes,
                    'policy_loss': float(last_update_metrics['policy_loss']),
                    'value_loss': float(last_update_metrics['value_loss']),
                    'entropy': float(-last_update_metrics['entropy_loss']),
                    'approx_kl': float(last_update_metrics['approx_kl']),
                    'clip_fraction': float(last_update_metrics['clip_fraction']),
                    'caps_loss': float(last_update_metrics.get('caps_loss', 0.0)),
                    'learning_rate': float(current_lr),
                    'ent_coef': float(current_ent),
                })
            save_log()

            # Spike detection and curriculum management
            pos_error_mean = eval_metrics['pos_error_mean']
            per_path_errors = {pt: m['pos_error_mean'] for pt, m in eval_metrics.get('per_path', {}).items()} \
                or {'circle': pos_error_mean}
            curriculum.record_eval(per_path_errors)

            # Spike detection (suppressed after curriculum transitions)
            if spike_suppression_evals > 0:
                spike_suppression_evals -= 1
            elif check_for_spike(pos_error_mean, prev_eval_error):
                print(f"\n  WARNING: Error spiked from {prev_eval_error*1000:.1f}mm to {pos_error_mean*1000:.1f}mm")
                spike_path = output_dir / f"spike_detected_{timestep + 1}.pt"
                agent.save(str(spike_path), obs_rms=obs_rms, reward_normalizer=reward_normalizer)
                print(f"  Saved spike checkpoint: {spike_path}")
            prev_eval_error = pos_error_mean

            # Curriculum advancement check
            if curriculum.should_advance():
                stage_params = curriculum.advance()

                # Suppress spike detection after transition (normalization change causes false spikes)
                spike_suppression_evals = 3

                # Notify scheduler of transition (triggers LR/entropy warmup)
                scheduler.on_curriculum_transition()
                boosted_lr = scheduler.get_lr(timestep)
                boosted_ent = scheduler.get_entropy_coef(timestep)

                print(f"\n{'='*60}")
                print(f"CURRICULUM: Advancing to Stage {curriculum.current_stage}")
                print(f"  Stage params: {stage_params}")
                print(f"  LR boosted: {boosted_lr:.2e} (warmup for {scheduler.warmup_steps} steps)")
                print(f"  Entropy boosted: {boosted_ent:.4f}")
                print('='*60)

                # Recreate environment with new stage parameters
                env.close()
                env, current_path_type = make_env_with_stage(config, stage_params, bidirectional=bidirectional)
                print(f"  Path types in pool: {stage_params.get('path_types', ['circle'])}")

                # Unfreeze normalizers to allow gradual adaptation (don't reset - too disruptive)
                obs_rms.unfreeze()
                reward_normalizer.unfreeze()

                # Reset observation
                obs, _ = env.reset()
                obs_rms.update(obs)
                obs = obs_rms.normalize(obs)

                # Set new warmup end point (normalizers unfrozen for gradual adaptation)
                warmup_end_step = timestep + curriculum.warmup_steps
                print(f"  Normalizers unfrozen. Will re-freeze at step {warmup_end_step}")

        # Save checkpoint
        if (timestep + 1) % config.training.save_frequency == 0:
            checkpoint_path = output_dir / f"checkpoint_{timestep + 1}.pt"
            agent.save(str(checkpoint_path), obs_rms=obs_rms, reward_normalizer=reward_normalizer)
            print(f"\n  Saved checkpoint: {checkpoint_path}")

    # Final save
    final_path = output_dir / "final_model.pt"
    agent.save(str(final_path), obs_rms=obs_rms, reward_normalizer=reward_normalizer)

    # Save final log
    training_log['end_time'] = datetime.now().isoformat()
    training_log['total_episodes'] = num_episodes
    save_log()

    print(f"\n{'='*60}")
    print(f"Training complete! Final model saved to: {final_path}")
    print(f"Training log saved to: {log_path}")
    print('='*60)

    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default=str(project_root / "configs" / "circle_baseline.yaml"))
    args = parser.parse_args()

    config_path = Path(args.config)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    train(config)
