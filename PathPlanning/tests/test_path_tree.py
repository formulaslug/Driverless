import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from path_tree import find_nearest_waypoints, get_path_tree

# Simple test patterns
def get_test_setup():
    # simple track
    simple_cones = np.array([
        [0, 0], [10, 0], [20, 2], [30, 5],
        [0, 5], [10, 5], [20, 7], [30, 10]
    ])

    # vehicle position and heading
    vehicle_pos = np.array([5, 2.5])
    vehicle_heading = 0  # facing right (0 radians)

    return simple_cones, vehicle_pos, vehicle_heading

# Test find_nearest_waypoints functionality
def test_find_nearest():
    print("Testing find_nearest_waypoints...")

    # create some test waypoints
    waypoints = np.array([
        [5, 0], [10, 0], [15, 0],  # forward waypoints
        [5, 5], [10, 5], [15, 5],  # forward waypoints
        [-5, 0], [-10, 0]  # backward waypoints (should be filtered)
    ])

    vehicle_pos = np.array([0, 2.5])
    vehicle_heading = 0  # facing right

    # find k=3 nearest forward waypoints
    k = 3
    nearest = find_nearest_waypoints(vehicle_pos, vehicle_heading, waypoints, k)

    # check that we got k waypoints
    assert len(nearest) == k, f"Expected {k} waypoints, got {len(nearest)}"

    # check that all waypoints are tuples
    assert all(isinstance(wp, tuple) for wp in nearest), "Waypoints not tuples"

    # check that backward waypoints were filtered out
    for wp in nearest:
        assert wp[0] > 0, "Backward waypoint was not filtered"

    print(f"  ✓ Found {len(nearest)} nearest waypoints (forward only)")
    print("✓ find_nearest_waypoints tests passed\n")

# Test get_path_tree functionality
def test_path_tree():
    print("Testing get_path_tree...")

    cones, vehicle_pos, vehicle_heading = get_test_setup()

    # test with small depth first
    max_depth = 2
    k_start = 3

    paths = get_path_tree(cones, vehicle_pos, vehicle_heading, max_depth, k_start)

    # check that paths were generated
    assert len(paths) > 0, "No paths generated"

    # check that all paths are lists
    assert all(isinstance(path, list) for path in paths), "Paths not lists"

    # check that paths contain waypoint tuples
    for path in paths[:5]:  # check first 5 paths
        assert all(isinstance(wp, tuple) for wp in path), "Path waypoints not tuples"
        assert all(len(wp) == 2 for wp in path), "Waypoints not 2D"

    print(f"  ✓ Generated {len(paths)} paths with max_depth={max_depth}")

    # test with different depth
    max_depth = 3
    paths_deep = get_path_tree(cones, vehicle_pos, vehicle_heading, max_depth, k_start)
    assert len(paths_deep) >= len(paths), "Deeper tree should have more paths"
    print(f"  ✓ Generated {len(paths_deep)} paths with max_depth={max_depth}")

    print("✓ get_path_tree tests passed\n")

# Test execution speed
def test_speed():
    print("Testing path_tree execution speed...")

    cones, vehicle_pos, vehicle_heading = get_test_setup()

    # test find_nearest_waypoints speed
    from delaunay import get_midpoints
    waypoints, _, _ = get_midpoints(cones)

    iterations = 100
    k = 5

    start = time.time()
    for _ in range(iterations):
        nearest = find_nearest_waypoints(vehicle_pos, vehicle_heading, waypoints, k)
    end = time.time()

    avg_time_ms = ((end - start) / iterations) * 1000
    print(f"  find_nearest_waypoints: {avg_time_ms:.2f}ms avg")

    # test get_path_tree speed
    iterations = 10  # fewer iterations since this is slower
    max_depth = 2
    k_start = 3

    start = time.time()
    for _ in range(iterations):
        paths = get_path_tree(cones, vehicle_pos, vehicle_heading, max_depth, k_start)
    end = time.time()

    avg_time_ms = ((end - start) / iterations) * 1000
    freq_hz = 1000 / avg_time_ms

    print(f"  get_path_tree (depth={max_depth}): {avg_time_ms:.2f}ms avg, {freq_hz:.0f} Hz")

    print("✓ Speed tests complete\n")

# Test forward-only filtering
def test_forward_filtering():
    print("Testing forward-only filtering...")

    # create waypoints in all directions
    waypoints = np.array([
        [10, 0],   # directly ahead
        [10, 10],  # forward right
        [0, 10],   # directly right
        [-10, 10], # backward right
        [-10, 0],  # directly behind
        [-10, -10],# backward left
        [0, -10],  # directly left
        [10, -10]  # forward left
    ])

    vehicle_pos = np.array([0, 0])
    vehicle_heading = 0  # facing right (positive x direction)
    k = 10  # try to get all

    nearest = find_nearest_waypoints(vehicle_pos, vehicle_heading, waypoints, k)

    # should only get forward waypoints (±90° from heading)
    # heading = 0 means forward is positive x direction
    # so we should get waypoints with x > 0
    for wp in nearest:
        assert wp[0] > 0, f"Non-forward waypoint included: {wp}"

    print(f"  ✓ Correctly filtered to {len(nearest)} forward waypoints out of {len(waypoints)}")
    print("✓ Forward filtering tests passed\n")

if __name__ == "__main__":
    test_find_nearest()
    test_path_tree()
    test_speed()
    test_forward_filtering()
    print("All path_tree tests passed!")
