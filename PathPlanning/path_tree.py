"""
Path Tree Population Module
Deadline: Nov 22, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import config
from delaunay import get_midpoints

# Need vehicle_heading in RADIANS from localization
# Find k nearest waypoints in the forward arc (±90°) of the vehicle
def find_nearest_waypoints(vehicle_pos, vehicle_heading, waypoints, k):
    # compute vector from vehicle position to each waypoint
    vectors = waypoints - vehicle_pos
    
    # compute angles of each waypoint using arctan2 for 4 quadrant calc
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diff = angles - vehicle_heading
    
    # normalize angles to the range [-pi, pi]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    
    # create boolean array to keep all waypoints that are in front of the car and apply filter
    forward_mask = np.abs(angle_diff) < (np.pi / 2)
    masked_waypoints = waypoints[forward_mask]
    
    # compute the magnitude of each vector using np.linalg.norm and get k shortest distances
    distances = np.linalg.norm(masked_waypoints - vehicle_pos, axis=1)
    k_indices = np.argsort(distances)[:k]
    
    # return the k nearest waypoints
    k_waypoints = masked_waypoints[k_indices]
    return [tuple(np.round(wp, decimals=6)) for wp in k_waypoints]

# Generate path tree from cone positions (main API function)
def get_path_tree(cones, vehicle_pos, vehicle_heading, max_depth, k_start):
    # get the possible waypoints
    waypoints, waypoint_graph, tri = get_midpoints(cones)

    # get the k nearest waypoints
    starting_waypoints = find_nearest_waypoints(vehicle_pos, vehicle_heading, waypoints, k_start)

    # Helper function to check if waypoint is forward of vehicle
    def is_forward(waypoint):
        # Check if waypoint is in forward half-plane from vehicle
        vec = np.array(waypoint) - np.array(vehicle_pos)
        # Gets vector in the direction of the vehicle
        forward_dir = np.array([np.cos(vehicle_heading), np.sin(vehicle_heading)])
        # If the two vectors are in similar direction, their dot product will be > 0
        return np.dot(vec, forward_dir) > 0

    current_level = [[wp] for wp in starting_waypoints]
    for _ in range(max_depth):
        next_level = []
        for path in current_level:
            last = path[-1]
            for next_wp in waypoint_graph.get(last, set()):
                # Only add waypoints that are forward AND not already visited
                if next_wp not in path and is_forward(next_wp):
                    next_level.append(path + [next_wp])

        current_level = next_level

    return current_level

# Visualize the paths with cones and vehicle position
def visualize_paths(paths, cones, vehicle_pos, vehicle_heading, max_paths=None):

    plt.figure(figsize=(14, 10))

    # Convert to numpy array for easier plotting
    cones = np.array(cones)
    vehicle_pos = np.array(vehicle_pos)

    # Get all possible waypoints from Delaunay triangulation
    waypoints, _, _ = get_midpoints(cones)

    # Filter to only forward-facing waypoints (±90° arc)
    vectors = waypoints - vehicle_pos
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diff = angles - vehicle_heading
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    forward_mask = np.abs(angle_diff) < (np.pi / 2)
    masked_waypoints = waypoints[forward_mask]

    # Plot only masked waypoints (small but visible)
    plt.scatter(masked_waypoints[:, 0], masked_waypoints[:, 1], c='lightgray', s=30,
                marker='o', edgecolors='gray', linewidth=0.5,
                label='Available Waypoints', zorder=2, alpha=0.7)

    # Plot cones
    plt.scatter(cones[:, 0], cones[:, 1], c='red', s=150,
                marker='o', edgecolors='darkred', linewidth=2,
                label='Cones', zorder=5, alpha=0.8)

    # Plot vehicle position
    plt.scatter(vehicle_pos[0], vehicle_pos[1], c='blue', s=500,
                marker='*', edgecolors='black', linewidth=2,
                label='Vehicle', zorder=10)

    # Plot vehicle heading arrow
    arrow_length = 0.8
    dx = arrow_length * np.cos(vehicle_heading)
    dy = arrow_length * np.sin(vehicle_heading)
    plt.arrow(vehicle_pos[0], vehicle_pos[1], dx, dy,
              head_width=0.25, head_length=0.2, fc='blue', ec='black',
              linewidth=1.5, zorder=10)

    # Randomly sample paths to display if specified
    if max_paths and len(paths) > max_paths:
        paths_to_show = random.sample(paths, max_paths)
    else:
        paths_to_show = paths

    # Plot all paths with different colors and styles
    colors = ['green', 'orange', 'purple', 'cyan', 'magenta',
              'lime', 'pink', 'gold', 'brown', 'navy']
    line_styles = ['-', '--', '-.', ':']  # Solid, dashed, dash-dot, dotted

    for i, path in enumerate(paths_to_show):
        if len(path) == 0:
            continue

        # Convert path to numpy array for plotting
        path_array = np.array(path)
        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]

        # Plot path as connected line with style variation
        plt.plot(path_array[:, 0], path_array[:, 1],
                color=color, linewidth=2.5, alpha=0.8, linestyle=linestyle,
                marker='o', markersize=7, markeredgecolor='black',
                markeredgewidth=0.8, zorder=3)

        # Add path number at the end of each path
        end_point = path_array[-1]
        plt.text(end_point[0], end_point[1], f' {i+1}',
                fontsize=10, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=color, alpha=0.8), zorder=15)

        # Highlight starting waypoint with larger marker
        start_point = path_array[0]
        plt.scatter(start_point[0], start_point[1],
                   c=color, s=200, marker='s', edgecolors='black',
                   linewidth=2, zorder=8, alpha=0.9)

    # Add path count to legend
    if len(paths) > 0:
        if max_paths and len(paths) > max_paths:
            label_text = f'{len(paths_to_show)}/{len(paths)} paths shown'
        else:
            label_text = f'{len(paths)} paths generated'
        plt.plot([], [], color='gray', linewidth=2.5, label=label_text)

    plt.xlabel('X position (m)', fontsize=13, fontweight='bold')
    plt.ylabel('Y position (m)', fontsize=13, fontweight='bold')
    plt.title('Path Planning Visualization', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    oval_track = [
        # Outer boundary
        [0, 0], [2, -1], [4, -1.5], [6, -1.5], [8, -1], [10, 0],
        [10, 2], [8, 3], [6, 3.5], [4, 3.5], [2, 3], [0, 2],
        # Inner boundary
        [2, 0.5], [4, 0], [6, 0], [8, 0.5], [8, 1.5], [6, 2],
        [4, 2], [2, 1.5]
    ]

    vehicle_pos = [3, -0.5]  # On racing line between inner/outer cones
    vehicle_heading = np.radians(0)  # Heading straight down the track

    paths = get_path_tree(
        cones=oval_track,
        vehicle_pos=vehicle_pos,
        vehicle_heading=vehicle_heading,
        max_depth=4,
        k_start=2
    )

    visualize_paths(paths, oval_track, vehicle_pos, vehicle_heading, max_paths=10)
