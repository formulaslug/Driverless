import numpy as np
from numpy.typing import ArrayLike
from delaunay import get_midpoints
from typing import List, Tuple

def find_nearest_waypoints(vehicle_pos: ArrayLike, vehicle_heading: float, waypoints: np.ndarray, k: int) -> List[Tuple[float, float]]:
    """
    Find k nearest waypoints in the forward arc (±90°) of the vehicle.

    vehicle_heading must be in RADIANS.
    """
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

def get_path_tree(cones: ArrayLike, vehicle_pos: ArrayLike, vehicle_heading: float, max_depth: int, k_start: int) -> List[List[Tuple[float, float]]]:
    """
    Generate breadth-first tree of possible paths through waypoint graph.

    Returns list of paths, where each path is a list of waypoint tuples.
    """
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

    # create starting level from selected waypoints
    current_level = [[wp] for wp in starting_waypoints]
    
    # iterate in range of max depth to get possible paths
    for _ in range(max_depth):
        next_level = []
        for path in current_level:
            last = path[-1]     # get the last coordinate in the path
            for next_wp in waypoint_graph.get(last, set()):
                # Only add waypoints that are forward AND not already visited
                if next_wp not in path and is_forward(next_wp):
                    next_level.append(path + [next_wp])
                    
        # we should prune here and cut the list down
        current_level = next_level
    
    # here we should be selecting and returning the best path
    return current_level
