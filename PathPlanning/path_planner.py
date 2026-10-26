import logging
import numpy as np
from scipy.spatial import QhullError
from numpy.typing import ArrayLike
from typing import Tuple, Optional
from path_tree import get_path_tree
from beam_search import beam_search_prune
from path_smoother import smooth_path
import config as cfg

logger = logging.getLogger(__name__)

def plan_path(
    cones: ArrayLike,
    coordinate_confidence: ArrayLike,
    colors: ArrayLike,
    vehicle_pos: ArrayLike,
    vehicle_heading: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Complete path planning pipeline

    Args:
        cones: Cone positions (n_cones, 2) - [x, y] in vehicle frame
        coordinate_confidence: Positional uncertainty radius per cone (n_cones,) in meters
        colors: Cone color probabilities (n_cones, 4) - [blue, yellow, orange_small, orange_large]
        vehicle_pos: Vehicle position [x, y]
        vehicle_heading: Vehicle heading in radians

    Returns:
        Tuple of (smooth_path, curvature):
            - smooth_path: np.ndarray (n_points, 2) or None if no path found
            - curvature: np.ndarray (n_points,) or None if no path found
    """
    if len(cones) < cfg.MIN_CONES_FOR_VALID_PATH:
        return None, None
    
    # generate all possible paths with level-by-level beam search pruning
    try:
        candidate_paths = get_path_tree(
            cones, coordinate_confidence, colors, vehicle_pos, vehicle_heading,
            cfg.MAX_TREE_DEPTH, cfg.K_START
        )
    except QhullError:
        logger.warning("Delaunay triangulation failed (cones degenerate or collinear)")
        return None, None

    # need at least 4 waypoints for cubic spline
    candidate_paths = [path for path in candidate_paths if len(path) >= 4]
    if len(candidate_paths) == 0:
        return None, None

    # select best path from candidates
    best_path = beam_search_prune(candidate_paths, cones, coordinate_confidence, colors, beam_width=1)[0]

    # start the path at the vehicle
    waypoints = np.array(best_path)
    if np.linalg.norm(waypoints[0] - np.asarray(vehicle_pos, float)) > 1e-6:
        waypoints = np.vstack([np.asarray(vehicle_pos, float), waypoints])

    # smooth the path and calculate curvature
    try:
        smooth_coords, curvature = smooth_path(waypoints, num_points=cfg.SPLINE_NUM_POINTS)
    except (ValueError, TypeError):
        logger.warning("Path smoothing failed", exc_info=True)
        return None, None

    return smooth_coords, curvature
