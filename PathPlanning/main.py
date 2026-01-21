import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, Optional
from path_tree import get_path_tree
from beam_search import beam_search_prune
from path_smoother import smooth_path
import config as cfg

def plan_path(
    cones: ArrayLike,
    colors: ArrayLike,
    vehicle_pos: ArrayLike,
    vehicle_heading: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Complete path planning pipeline for autonomous racing.

    Args:
        cones: Cone positions (n_cones, 2) - [x, y] in vehicle frame
        colors: Cone color probabilities (n_cones, 4) - [blue, yellow, orange_small, orange_large]
        vehicle_pos: Vehicle position [x, y]
        vehicle_heading: Vehicle heading in radians

    Returns:
        Tuple of (smooth_path, curvature):
            - smooth_path: np.ndarray (n_points, 2) or None if no path found
            - curvature: np.ndarray (n_points,) or None if no path found
    """
    # generate all possible paths with level-by-level beam search pruning
    candidate_paths = get_path_tree(
        cones, colors, vehicle_pos, vehicle_heading,
        cfg.MAX_TREE_DEPTH, cfg.K_START
    )

    # handle no paths found
    if len(candidate_paths) == 0:
        return None, None

    # select best path from candidates
    best_paths = beam_search_prune(candidate_paths, cones, colors, beam_width=1)

    if len(best_paths) == 0:
        return None, None

    best_path = best_paths[0]

    # need at least 4 waypoints for cubic spline
    if len(best_path) < 4:
        return None, None

    # smooth the path and calculate curvature
    smooth_coords, curvature = smooth_path(np.array(best_path), num_points=cfg.SPLINE_NUM_POINTS)

    return smooth_coords, curvature
