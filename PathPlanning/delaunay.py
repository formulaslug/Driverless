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

def visualize(test_cases):
    # Visualize one or more tracks with triangulation and waypoint graph
    # test_cases: list of (name, cone_data) tuples

    n = len(test_cases)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
    axes = axes.flatten()

    for i, (name, cone_data) in enumerate(test_cases):
        ax = axes[i]
        points = np.array(cone_data)
        waypoints, waypoint_graph, tri = get_midpoints(cone_data)

        # Plot triangulation and cones
        ax.triplot(points[:, 0], points[:, 1], tri.simplices, 'b-', linewidth=0.5)
        ax.plot(points[:, 0], points[:, 1], 'ro', markersize=6, label="Cones")
        ax.plot(waypoints[:, 0], waypoints[:, 1], 'go', markersize=4, label="Waypoints")

        # Draw waypoint graph connections
        for wp, neighbors in waypoint_graph.items():
            for neighbor in neighbors:
                ax.plot([wp[0], neighbor[0]], [wp[1], neighbor[1]], 'g-', linewidth=0.8, alpha=0.4)

        ax.set_title(f'{name} ({len(points)} cones)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

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
          ("Random 30", random_scatter),
          ("Simple Track", simple_track),
          ("Slalom", slalom),
          ("Grid", grid),
          ("Oval Track", oval_track)
      ]

    for name, cone_data in test_cases:
          points = np.array(cone_data)

          start = time.perf_counter()
          waypoints, waypoint_graph, tri = get_midpoints(cone_data)
          elapsed = time.perf_counter() - start

          print(f"{name:15} | Cones: {len(points):3} | Time: {elapsed*1000:6.2f} ms | Hz: {1/elapsed:6.1f}")

    # Visualize all test tracks
    visualize(test_cases)
