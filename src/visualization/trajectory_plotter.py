"""3D trajectory visualization tools."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class TrajectoryPlotter:
    """Create 3D trajectory visualizations."""

    def plot_3d_trajectory(self, ee_positions, target_positions,
                          save_path=None, show=False):
        """Plot 3D trajectory comparison.

        Args:
            ee_positions: (T, 3) actual EE trajectory
            target_positions: (T, 3) reference path
            save_path: Optional path to save figure
            show: Whether to display interactively
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot reference path
        ax.plot(target_positions[:, 0],
                target_positions[:, 1],
                target_positions[:, 2],
                'b-', linewidth=2, label='Reference', alpha=0.7)

        # Plot actual trajectory
        ax.plot(ee_positions[:, 0],
                ee_positions[:, 1],
                ee_positions[:, 2],
                'r-', linewidth=1.5, label='Actual', alpha=0.8)

        # Markers every N steps
        stride = max(len(ee_positions) // 20, 1)
        ax.scatter(ee_positions[::stride, 0],
                  ee_positions[::stride, 1],
                  ee_positions[::stride, 2],
                  c='red', s=30, alpha=0.6, marker='o')

        # Start/end markers
        ax.scatter([ee_positions[0, 0]], [ee_positions[0, 1]], [ee_positions[0, 2]],
                  c='green', s=100, marker='*', label='Start', zorder=10)
        ax.scatter([ee_positions[-1, 0]], [ee_positions[-1, 1]], [ee_positions[-1, 2]],
                  c='orange', s=100, marker='X', label='End', zorder=10)

        # Formatting
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title('End-Effector Trajectory Tracking', fontsize=14)

        # Equal aspect ratio
        max_range = np.array([
            ee_positions[:, 0].max() - ee_positions[:, 0].min(),
            ee_positions[:, 1].max() - ee_positions[:, 1].min(),
            ee_positions[:, 2].max() - ee_positions[:, 2].min()
        ]).max() / 2.0
        mid = np.array([
            (ee_positions[:, 0].max() + ee_positions[:, 0].min()) / 2,
            (ee_positions[:, 1].max() + ee_positions[:, 1].min()) / 2,
            (ee_positions[:, 2].max() + ee_positions[:, 2].min()) / 2
        ])
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory plot saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_error_timeline(self, position_errors, dt, save_path=None, show=False):
        """Plot position error over time.

        Args:
            position_errors: (T,) array of position errors in meters
            dt: Timestep in seconds
            save_path: Optional path to save figure
            show: Whether to display interactively
        """
        fig, ax = plt.subplots(figsize=(10, 4))

        times = np.arange(len(position_errors)) * dt
        errors_mm = position_errors * 1000  # Convert to mm

        ax.plot(times, errors_mm, linewidth=1.5, label='Position Error')

        # Statistics
        mean_err = np.mean(errors_mm)
        max_err = np.max(errors_mm)
        rms_err = np.sqrt(np.mean(errors_mm ** 2))

        ax.axhline(mean_err, color='orange', linestyle='--',
                  label=f'Mean: {mean_err:.2f} mm')
        ax.axhline(max_err, color='red', linestyle=':',
                  label=f'Max: {max_err:.2f} mm')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Position Error (mm)', fontsize=12)
        ax.set_title('Tracking Error Over Time', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Error plot saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
