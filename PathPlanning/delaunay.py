from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
    
def create_delaunay_triangulation(cones):
    # Create a Delaunay Triangulation from the cone positions
    points = np.array(cones)
    
    return Delaunay(points)

def get_midpoint_graph(tri, points):
    # Get a midpoint graph from the Delaunay Triangulation
    waypoints = []
    
    for simplex in tri.simplices:
        i, j ,k = simplex
        # Get coordinates of each point
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # Compute midpoints of each edge and add to waypoints
        waypoints.append((p1 + p2) / 2)
        waypoints.append((p1 + p3) / 2)
        waypoints.append((p2 + p3) / 2)
       
    return np.array(waypoints)

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
    test_cone_data = oval_track# Change this to try different patterns

    # Parse and create triangulation
    points = np.array(test_cone_data)
    tri = create_delaunay_triangulation(test_cone_data)
    waypoints = get_midpoint_graph(tri, points)

    # Visualize
    visualize_triangulation(points, tri, waypoints)
