from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt

def parse_cone_data(cones):
    # Parses cone data from mapping team to create points to use for triangulation

    return np.array(cones)
    
    
def create_delaunay_triangulation(points):
    # Create a Delaunay Triangulation from the cone positions

    return Delaunay(points)

def visualize_triangulation(points, tri):
    # Visualize and verify triangulation of points

    plt.figure(figsize=(10,8))
    
    plt.triplot(points[ : , 0], points[ : , 1], tri.simplices, 'b-', linewidth=0.5)
    plt.plot(points[ : , 0], points[ : , 1], 'ro', markersize=10, label="Cones")
    
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
    test_cone_data = random_scatter# Change this to try different patterns

    # Parse and create triangulation
    points = parse_cone_data(test_cone_data)
    tri = create_delaunay_triangulation(points)

    # Visualize
    visualize_triangulation(points, tri)
