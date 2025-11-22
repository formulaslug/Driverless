from scipy.spatial import Delaunay
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import time 
    
def get_midpoints(cones):
    # Get a midpoint graph from the Delaunay Triangulation
    points = np.array(cones)    
    tri = Delaunay(points)
    waypoint_graph = defaultdict(set)
    
    # Vectorize this for speedup instead of using a loop
    # Get coordinates of each point
    p1 = points[tri.simplices[:,0]]
    p2 = points[tri.simplices[:,1]]
    p3 = points[tri.simplices[:,2]]
    
    # Compute midpoints of each edge and store
    wayp1p2 = ((p1 + p2) / 2)
    wayp1p3 = ((p1 + p3) / 2)
    wayp2p3 = ((p2 + p3) / 2)
    
    for i in range(len(wayp1p2)):
        wp1, wp2, wp3 = wayp1p2[i], wayp1p3[i], wayp2p3[i]
        # Convert to tuples (hashable for dict keys)
        wp1_tuple = tuple(np.round(wp1, decimals=6))
        wp2_tuple = tuple(np.round(wp2, decimals=6))
        wp3_tuple = tuple(np.round(wp3, decimals=6))
        # Each waypoint connects to the other 2 in the triangle
        waypoint_graph[wp1_tuple].add(wp2_tuple)
        waypoint_graph[wp1_tuple].add(wp3_tuple)
        waypoint_graph[wp2_tuple].add(wp1_tuple)
        waypoint_graph[wp2_tuple].add(wp3_tuple)
        waypoint_graph[wp3_tuple].add(wp1_tuple)
        waypoint_graph[wp3_tuple].add(wp2_tuple)
      
    # Stack all the waypoint arrays into one vertically
    waypoints = np.unique(np.vstack([wayp1p2, wayp1p3, wayp2p3]), axis=0)
        
    return waypoints, waypoint_graph, tri

def visualize_triangulation(points, tri, waypoints):
    # Visualize and verify triangulation of points

    plt.figure(figsize=(10,8))

    plt.triplot(points[ : , 0], points[ : , 1], tri.simplices, 'b-', linewidth=0.5)
    plt.plot(points[ : , 0], points[ : , 1], 'ro', markersize=10, label="Cones")
    plt.plot(waypoints[ : , 0], waypoints[ : , 1], 'go', markersize=10, label="Waypoints")

    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Delaunay Triangulation of Cones')
    plt.legend()
    plt.show()

def visualize_waypoint_graph(waypoint_graph, points):
    # Visualize the waypoint graph connections in a separate window

    plt.figure(figsize=(10,8))

    # Draw all connections between waypoints
    for waypoint, neighbors in waypoint_graph.items():
        wx, wy = waypoint
        for neighbor in neighbors:
            nx, ny = neighbor
            # Draw line between connected waypoints
            plt.plot([wx, nx], [wy, ny], 'g-', linewidth=1, alpha=0.6)

    # Plot waypoints as nodes
    waypoint_coords = np.array(list(waypoint_graph.keys()))
    plt.plot(waypoint_coords[:, 0], waypoint_coords[:, 1], 'go',
             markersize=8, label="Waypoints", zorder=3)

    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Waypoint Graph Connections')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_combined(points, tri, waypoints, waypoint_graph):
    # Visualize both triangulation and waypoint graph side-by-side

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left plot: Delaunay Triangulation
    ax1.triplot(points[:, 0], points[:, 1], tri.simplices, 'b-', linewidth=0.5)
    ax1.plot(points[:, 0], points[:, 1], 'ro', markersize=10, label="Cones")
    ax1.plot(waypoints[:, 0], waypoints[:, 1], 'go', markersize=10, label="Waypoints")
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_title('Delaunay Triangulation of Cones')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Waypoint Graph
    # Draw all connections between waypoints
    for waypoint, neighbors in waypoint_graph.items():
        wx, wy = waypoint
        for neighbor in neighbors:
            nx, ny = neighbor
            ax2.plot([wx, nx], [wy, ny], 'g-', linewidth=1, alpha=0.6)

    # Plot waypoints as nodes
    waypoint_coords = np.array(list(waypoint_graph.keys()))
    ax2.plot(waypoint_coords[:, 0], waypoint_coords[:, 1], 'go',
             markersize=8, label="Waypoints", zorder=3)

    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    ax2.set_title('Waypoint Graph Connections')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # Test data option 1: Simple track-like pattern
    simple_track = [
        [0, 0], [1, 0], [2, 0.5], [3, 1], [4, 1.5],
        [4, -0.5], [3, -1], [2, -1.5], [1, -1], [0, -0.5]
    ]

    # Test data option 2: Oval/loop track (more realistic)
    oval_track = [
        # Outer boundary
        [0, 0], [2, -1], [4, -1.5], [6, -1.5], [8, -1], [10, 0],
        [10, 2], [8, 3], [6, 3.5], [4, 3.5], [2, 3], [0, 2],
        # Inner boundary
        [2, 0.5], [4, 0], [6, 0], [8, 0.5], [8, 1.5], [6, 2],
        [4, 2], [2, 1.5]
    ]

    # Test data option 3: Slalom/chicane pattern
    slalom = [
        [0, 0], [1, 1], [2, -1], [3, 1.5], [4, -1.5],
        [5, 2], [6, -2], [7, 2.5], [8, -2.5], [9, 0],
        [0, -0.5], [1, -2], [2, 1], [3, -2.5], [4, 2],
        [5, -3], [6, 2.5], [7, -3.5], [8, 3], [9, 0.5]
    ]

    # Test data option 4: Grid pattern (stress test)
    grid = []
    for x in range(0, 10, 2):
        for y in range(-4, 5, 2):
            grid.append([x, y])

    # Test data option 5: Random scattered (realistic chaos)
    import random
    random.seed(42)
    random_scatter = [[random.uniform(0, 10), random.uniform(-3, 3)] for _ in range(30)]

    # Choose which test data to use
    test_cases = [
          ("Simple Track", simple_track),
          ("Slalom", slalom),
          ("Grid", grid),
          ("Random 30", random_scatter),
          ("Oval Track", oval_track)
      ]

    for name, cone_data in test_cases:
          points = np.array(cone_data)

          start = time.perf_counter()
          waypoints, waypoint_graph, tri = get_midpoints(cone_data)
          elapsed = time.perf_counter() - start

          print(f"{name:15} | Cones: {len(points):3} | Time: {elapsed*1000:6.2f} ms | Hz: {1/elapsed:6.1f}")

    # Visualize the last test case - both plots side-by-side
    visualize_combined(points, tri, waypoints, waypoint_graph)
