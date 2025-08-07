import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import Normalize

# Set English fonts and clean plotting
plt.rcParams["font.family"] = "DejaVu Sans"  # Use standard English font
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display


def plot_3d_trajectories(self_pos, oppo_pos, fire, lock, dir, file_name):
    ax = plt.figure().add_subplot(projection='3d')
    self_pos = np.array(self_pos)
    oppo_pos = np.array(oppo_pos)

    num_points = len(self_pos)

    self_colors = plt.cm.Blues(np.linspace(0.3, 1, num_points))
    oppo_colors = plt.cm.Reds(np.linspace(0.3, 1, num_points))
    min_z = min(np.min(self_pos[:, 1]), np.min(oppo_pos[:, 1]))

    # Plot projections on XOY plane
    ax.plot(self_pos[:, 0], self_pos[:, 2], zs=min_z - 1000, zdir='z', color=self_colors[num_points // 2],
            linestyle='--', linewidth=2, alpha=1)
    ax.plot(oppo_pos[:, 0], oppo_pos[:, 2], zs=min_z - 1000, zdir='z', color=oppo_colors[num_points // 2],
            linestyle='--', linewidth=2, alpha=1)

    for i in range(num_points - 1):
        ax.plot(oppo_pos[i:i + 2, 0], oppo_pos[i:i + 2, 2], oppo_pos[i:i + 2, 1], color=oppo_colors[i],
                label='opponent', zorder=2, linewidth=2)
        if i % 100 == 0:
            # Draw dashed line for opponent position projection
            ax.plot([oppo_pos[i, 0], oppo_pos[i, 0]],
                    [oppo_pos[i, 2], oppo_pos[i, 2]],
                    [oppo_pos[i, 1], min_z - 1000],
                    color=oppo_colors[i], linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    for i in range(num_points - 1):
        ax.plot(self_pos[i:i + 2, 0], self_pos[i:i + 2, 2], self_pos[i:i + 2, 1], color=self_colors[i], label='agent',
                zorder=2, linewidth=2)
        if i % 100 == 0:
            # Draw dashed line for agent position projection
            ax.plot([self_pos[i, 0], self_pos[i, 0]],
                    [self_pos[i, 2], self_pos[i, 2]],
                    [self_pos[i, 1], min_z - 1000],
                    color=self_colors[i], linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    for i in lock:
        ax.plot(self_pos[i:i + 2, 0], self_pos[i:i + 2, 2], self_pos[i:i + 2, 1], color='#FFDFBF', zorder=3,
                linewidth=0.8, alpha=1)

    # Create Normalize objects for color mapping
    self_norm = Normalize(vmin=0, vmax=num_points)
    oppo_norm = Normalize(vmin=0, vmax=num_points)

    # Create ScalarMappable objects for colorbars
    self_sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=self_norm)
    oppo_sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=oppo_norm)

    # Create colorbar axes
    self_cbar_ax = ax.inset_axes([1.15, 0.08, 0.03, 0.8])
    oppo_cbar_ax = ax.inset_axes([1.18, 0.08, 0.03, 0.8])

    # Add colorbars
    self_cbar = plt.colorbar(self_sm, cax=self_cbar_ax, label='Step number (agent)')
    oppo_cbar = plt.colorbar(oppo_sm, cax=oppo_cbar_ax, label='Step number (opponent)')

    # Set colorbar properties
    self_cbar.ax.yaxis.set_label_coords(-2, 0.5)
    self_cbar.set_ticks([])
    oppo_cbar.set_ticks(np.arange(0, num_points, 500))

    # Mark fire events
    for i in fire:
        ax.scatter(self_pos[i, 0], self_pos[i, 2], self_pos[i, 1], color='red', marker='^', s=8,
                   zorder=5 if i == fire[0] else 4)

    # Set labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Z Position (m)')
    ax.set_zlabel('Y Position (Altitude, m)')
    plt.title('3D Aircraft Trajectories', fontsize=14)

    # Save plot
    dir = os.path.join(dir, file_name)
    plt.savefig(dir, bbox_inches='tight', dpi=300)
    plt.close()


def plot_distance(distance, lock, missile, fire, dir, file_name):
    # Create figure
    plt.figure(figsize=(10, 7))

    # Plot distance curve
    plt.plot(distance, color='blue', linewidth=3, marker='None', zorder=2)

    # Highlight lock periods in light yellow
    for i in range(len(lock)):
        plt.axvspan(lock[i], lock[i] + 1, facecolor='#FFDFBF', alpha=0.6, zorder=1)

    # Highlight missile available periods in light blue
    for j in range(len(missile)):
        plt.axvspan(missile[j], missile[j] + 1, facecolor='#ACD8E6', alpha=0.6, zorder=1)

    # Mark fire events with red triangles
    for fire_step in fire:
        plt.scatter(fire_step, distance[fire_step], color='red', s=60, marker='^', zorder=3)

    # Formatting
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Step number', fontsize=25)
    plt.ylabel('Distance from opponent (m)', fontsize=25)
    plt.title('Distance vs Time', fontsize=28)
    plt.grid(True, alpha=0.3)

    # Save plot
    dir = os.path.join(dir, file_name)
    plt.savefig(dir, dpi=300, bbox_inches='tight')
    plt.close()


def plot_2d_trajectories(ally_pos, enemy_pos, save_path=None):
    # Extract x and z coordinates
    ally_x, ally_y, ally_z = zip(*ally_pos) if ally_pos else ([], [], [])
    enemy_x, enemy_y, enemy_z = zip(*enemy_pos) if enemy_pos else ([], [], [])

    # Create figure
    plt.figure(figsize=(10, 10))

    # Plot trajectories
    plt.plot(ally_x, ally_z, 'b-o', label='Agent Trajectory', markersize=1)
    plt.plot(enemy_x, enemy_z, 'r-o', label='Enemy Trajectory', markersize=1)

    # Formatting
    plt.legend()
    plt.title('Aircraft Trajectories (Top View)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Save if path provided
    if save_path:
        folder_path = os.path.dirname(save_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()