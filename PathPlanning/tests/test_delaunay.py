import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from delaunay import get_midpoints

# Simple test cone patterns
def get_test_cones():
    # simple square pattern
    simple = np.array([
        [0, 0], [10, 0], [10, 10], [0, 10]
    ])

    # oval track with inner and outer boundaries
    oval = np.array([
        # outer boundary
        [0, 0], [20, 0], [40, 5], [50, 15], [50, 35], [40, 45], [20, 50], [0, 50], [-10, 40], [-10, 10],
        # inner boundary
        [10, 15], [25, 15], [35, 20], [35, 30], [25, 35], [10, 35], [5, 25]
    ])

    # small test for speed
    small = np.array([[0, 0], [5, 0], [5, 5], [0, 5], [2.5, 2.5]])

    return {'simple': simple, 'oval': oval, 'small': small}

# Test basic functionality
def test_functionality():
    print("Testing get_midpoints functionality...")
    test_cones = get_test_cones()

    for name, cones in test_cones.items():
        waypoints, graph, tri = get_midpoints(cones)

        # check that waypoints were generated
        assert len(waypoints) > 0, f"{name}: No waypoints generated"

        # check that graph has entries
        assert len(graph) > 0, f"{name}: Graph is empty"

        # check that triangulation worked
        assert tri.simplices.shape[0] > 0, f"{name}: No triangles created"

        # check waypoints are 2D coordinates
        assert waypoints.shape[1] == 2, f"{name}: Waypoints not 2D"

        print(f"  {name}: {len(cones)} cones -> {len(waypoints)} waypoints, {len(graph)} graph nodes")

    print("✓ Functionality tests passed\n")

# Test execution speed
def test_speed():
    print("Testing get_midpoints execution speed...")
    test_cones = get_test_cones()

    # run multiple iterations for accurate timing
    iterations = 100

    for name, cones in test_cones.items():
        start = time.time()
        for _ in range(iterations):
            waypoints, graph, tri = get_midpoints(cones)
        end = time.time()

        avg_time_ms = ((end - start) / iterations) * 1000
        freq_hz = 1000 / avg_time_ms

        print(f"  {name} ({len(cones)} cones): {avg_time_ms:.2f}ms avg, {freq_hz:.0f} Hz")

    print("✓ Speed tests complete\n")

# Test edge cases
def test_edge_cases():
    print("Testing edge cases...")

    # minimum triangulation (3 points)
    min_cones = np.array([[0, 0], [1, 0], [0, 1]])
    waypoints, graph, tri = get_midpoints(min_cones)
    assert len(waypoints) == 3, "Minimum case failed"
    print("  ✓ Minimum case (3 cones)")

    # nearly collinear points (small offset)
    nearly_collinear = np.array([[0, 0], [1, 1], [2, 2.1], [3, 3]])
    waypoints, graph, tri = get_midpoints(nearly_collinear)
    assert len(waypoints) > 0, "Nearly collinear case failed"
    print("  ✓ Nearly collinear points")

    print("✓ Edge case tests passed\n")

if __name__ == "__main__":
    test_functionality()
    test_speed()
    test_edge_cases()
    print("All delaunay tests passed!")
