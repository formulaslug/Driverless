import numpy as np
from numpy.typing import ArrayLike
from delaunay import get_midpoints
from typing import List, Tuple

def is_forward(waypoint: ArrayLike, vehicle_pos: ArrayLike, vehicle_heading: float) -> bool:
    """
    Check if waypoint(s) are in forward half-plane (±90°) from vehicle.

    Uses dot product: if vec · forward_dir > 0, waypoint is forward.
    vehicle_heading must be in RADIANS.
    """
    vec = np.array(waypoint) - np.array(vehicle_pos)
    forward_dir = np.array([np.cos(vehicle_heading), np.sin(vehicle_heading)])
    return np.dot(vec, forward_dir) > 0

def find_nearest_waypoints(vehicle_pos: ArrayLike, vehicle_heading: float, waypoints: np.ndarray, k: int) -> List[Tuple[float, float]]:
    """
    Find k nearest waypoints in the forward arc (±90°) of the vehicle.

    vehicle_heading must be in RADIANS.
    """
    # compute vector from vehicle position to each waypoint
    vectors = waypoints - vehicle_pos

    # get forward direction vector
    forward_dir = np.array([np.cos(vehicle_heading), np.sin(vehicle_heading)])

    # check which waypoints are forward using dot product
    dot_products = np.dot(vectors, forward_dir)
    forward_mask = dot_products > 0
    masked_waypoints = waypoints[forward_mask]

    # compute the magnitude of each vector using np.linalg.norm and get k shortest distances
    distances = np.linalg.norm(masked_waypoints - vehicle_pos, axis=1)
    k_indices = np.argsort(distances)[:k]

    # return the k nearest waypoints
    k_waypoints = masked_waypoints[k_indices]
    return [tuple(np.round(wp, decimals=6)) for wp in k_waypoints]

def get_path_tree(cones: ArrayLike, colors: ArrayLike, vehicle_pos: ArrayLike, vehicle_heading: float, max_depth: int, k_start: int) -> List[List[Tuple[float, float]]]:
    """
    Generate breadth-first tree of possible paths through waypoint graph.

    Returns list of paths, where each path is a list of waypoint tuples.
    """
    # get the possible waypoints
    waypoints, waypoint_graph, tri = get_midpoints(cones, colors)

    # get the k nearest waypoints
    starting_waypoints = find_nearest_waypoints(vehicle_pos, vehicle_heading, waypoints, k_start)

    # create starting level from selected waypoints
    current_level = [[wp] for wp in starting_waypoints]

    # iterate in range of max depth to get possible paths
    for _ in range(max_depth):
        next_level = []
        for path in current_level:
            last = path[-1]     # get the last coordinate in the path
            for next_wp in waypoint_graph.get(last, set()):
                # only add waypoints that are forward AND not already visited
                if next_wp not in path and is_forward(next_wp, vehicle_pos, vehicle_heading):
                    next_level.append(path + [next_wp])
                    
        # we should prune here and cut the list down
        current_level = next_level
    
    # here we should be selecting and returning the best path
    return current_level
