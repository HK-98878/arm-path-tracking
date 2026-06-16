"""3D trajectory visualization tools."""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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

    def plot_error_timeline(self, position_errors, dt, orientation_errors=None,
                            save_path=None, show=False):
        """Plot position (and optionally orientation) error over time.

        Args:
            position_errors: (T,) array of position errors in meters
            dt: Timestep in seconds
            orientation_errors: Optional (T,) array of geodesic orientation errors in radians
            save_path: Optional path to save figure
            show: Whether to display interactively
        """
        has_orientation = orientation_errors is not None and len(orientation_errors) > 0

        if has_orientation:
            fig, (ax_pos, ax_ori) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        else:
            fig, ax_pos = plt.subplots(figsize=(10, 4))

        times = np.arange(len(position_errors)) * dt
        errors_mm = position_errors * 1000  # Convert to mm

        ax_pos.plot(times, errors_mm, linewidth=1.5, label='Position Error', color='tab:blue')

        # Statistics
        mean_err = np.mean(errors_mm)
        max_err = np.max(errors_mm)

        ax_pos.axhline(mean_err, color='orange', linestyle='--',
                  label=f'Mean: {mean_err:.2f} mm')
        ax_pos.axhline(max_err, color='red', linestyle=':',
                  label=f'Max: {max_err:.2f} mm')

        ax_pos.set_ylabel('Position Error (mm)', fontsize=12)
        ax_pos.set_title('Tracking Error Over Time', fontsize=14)
        ax_pos.legend()
        ax_pos.grid(True, alpha=0.3)

        if has_orientation:
            errors_deg = np.degrees(orientation_errors)
            mean_ori = np.mean(errors_deg)
            max_ori = np.max(errors_deg)

            ax_ori.plot(times, errors_deg, linewidth=1.5, label='Orientation Error', color='tab:green')
            ax_ori.axhline(mean_ori, color='orange', linestyle='--',
                      label=f'Mean: {mean_ori:.2f} deg')
            ax_ori.axhline(max_ori, color='red', linestyle=':',
                      label=f'Max: {max_ori:.2f} deg')

            ax_ori.set_xlabel('Time (s)', fontsize=12)
            ax_ori.set_ylabel('Orientation Error (deg)', fontsize=12)
            ax_ori.legend()
            ax_ori.grid(True, alpha=0.3)
        else:
            ax_pos.set_xlabel('Time (s)', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Error plot saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_3d_playback(self, ee_positions, target_positions, dt=0.01):
        """Interactive 3D trajectory playback with slider.

        Args:
            ee_positions: (T, 3) actual EE trajectory
            target_positions: (T, 3) reference/target positions at each timestep
            dt: Timestep in seconds (for time display)
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Leave space at bottom for slider
        plt.subplots_adjust(bottom=0.15)

        n_steps = len(ee_positions)

        # Plot full reference path (faded)
        ax.plot(target_positions[:, 0],
                target_positions[:, 1],
                target_positions[:, 2],
                'b-', linewidth=1, label='Target path', alpha=0.3)

        # Plot full EE trajectory (faded)
        ax.plot(ee_positions[:, 0],
                ee_positions[:, 1],
                ee_positions[:, 2],
                'r-', linewidth=1, label='EE path', alpha=0.3)

        # Current position markers (will be updated)
        ee_marker, = ax.plot([ee_positions[0, 0]], [ee_positions[0, 1]], [ee_positions[0, 2]],
                            'ro', markersize=12, label='EE current')
        target_marker, = ax.plot([target_positions[0, 0]], [target_positions[0, 1]], [target_positions[0, 2]],
                                'b^', markersize=12, label='Target current')

        # Line connecting EE to target (error visualization)
        error_line, = ax.plot([ee_positions[0, 0], target_positions[0, 0]],
                             [ee_positions[0, 1], target_positions[0, 1]],
                             [ee_positions[0, 2], target_positions[0, 2]],
                             'g--', linewidth=2, alpha=0.7, label='Error')

        # Trail (recent history)
        trail_len = 50
        ee_trail, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.6)
        target_trail, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.6)

        # Start marker
        ax.scatter([ee_positions[0, 0]], [ee_positions[0, 1]], [ee_positions[0, 2]],
                  c='green', s=150, marker='*', label='Start', zorder=10)

        # Formatting
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)

        # Set axis limits based on full data
        all_pos = np.vstack([ee_positions, target_positions])
        max_range = np.array([
            all_pos[:, 0].max() - all_pos[:, 0].min(),
            all_pos[:, 1].max() - all_pos[:, 1].min(),
            all_pos[:, 2].max() - all_pos[:, 2].min()
        ]).max() / 2.0 * 1.1
        mid = np.array([
            (all_pos[:, 0].max() + all_pos[:, 0].min()) / 2,
            (all_pos[:, 1].max() + all_pos[:, 1].min()) / 2,
            (all_pos[:, 2].max() + all_pos[:, 2].min()) / 2
        ])
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        # Title with error info
        init_error = np.linalg.norm(ee_positions[0] - target_positions[0]) * 1000
        title = ax.set_title(f'Step 0/{n_steps-1} | Time: 0.00s | Error: {init_error:.1f}mm', fontsize=12)

        # Slider
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(ax_slider, 'Step', 0, n_steps - 1, valinit=0, valstep=1)

        def update(val):
            step = int(slider.val)

            # Update current markers
            ee_marker.set_data_3d([ee_positions[step, 0]], [ee_positions[step, 1]], [ee_positions[step, 2]])
            target_marker.set_data_3d([target_positions[step, 0]], [target_positions[step, 1]], [target_positions[step, 2]])

            # Update error line
            error_line.set_data_3d(
                [ee_positions[step, 0], target_positions[step, 0]],
                [ee_positions[step, 1], target_positions[step, 1]],
                [ee_positions[step, 2], target_positions[step, 2]]
            )

            # Update trails
            trail_start = max(0, step - trail_len)
            ee_trail.set_data_3d(
                ee_positions[trail_start:step+1, 0],
                ee_positions[trail_start:step+1, 1],
                ee_positions[trail_start:step+1, 2]
            )
            target_trail.set_data_3d(
                target_positions[trail_start:step+1, 0],
                target_positions[trail_start:step+1, 1],
                target_positions[trail_start:step+1, 2]
            )

            # Update title
            error = np.linalg.norm(ee_positions[step] - target_positions[step]) * 1000
            time = step * dt
            title.set_text(f'Step {step}/{n_steps-1} | Time: {time:.2f}s | Error: {error:.1f}mm')

            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()
        return fig
