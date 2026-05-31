"""Time-series metrics visualization tools."""

import matplotlib.pyplot as plt
import numpy as np


class MetricsPlotter:
    """Create time-series metric plots."""

    def plot_actions(self, actions, dt, save_path=None, show=False):
        """Plot 6D action trajectory.

        Args:
            actions: (T, 6) array
            dt: Timestep
            save_path: Optional path to save figure
            show: Whether to display interactively
        """
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        times = np.arange(len(actions)) * dt

        labels = ['Linear X', 'Linear Y', 'Linear Z',
                 'Angular X', 'Angular Y', 'Angular Z']

        for idx, (ax, label) in enumerate(zip(axes.flat, labels)):
            ax.plot(times, actions[:, idx], linewidth=1)
            ax.set_ylabel(f'{label}', fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Action: {label}')

        plt.suptitle('Policy Actions Over Time', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Actions plot saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_rewards(self, rewards, reward_components, dt,
                    save_path=None, show=False):
        """Plot reward breakdown over time.

        Args:
            rewards: (T,) array of total rewards
            reward_components: List of dicts with reward components
            dt: Timestep
            save_path: Optional path to save figure
            show: Whether to display interactively
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        times = np.arange(len(rewards)) * dt

        # Total reward
        axes[0].plot(times, rewards, linewidth=1.5, color='black')
        axes[0].set_ylabel('Total Reward', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Total Reward Over Time')

        # Components (stacked area or lines)
        # Extract component names from first element
        if len(reward_components) > 0:
            component_names = [k for k in reward_components[0].keys()
                             if k.startswith('r_')]

            for name in component_names:
                component_values = [rc.get(name, 0.0) for rc in reward_components]
                axes[1].plot(times, component_values, linewidth=1,
                           label=name, alpha=0.7)

        axes[1].set_xlabel('Time (s)', fontsize=12)
        axes[1].set_ylabel('Reward Components', fontsize=12)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Rewards plot saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
