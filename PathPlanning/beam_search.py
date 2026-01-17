import numpy as np
from typing import List, Tuple
from cost_functions import evaluate_path_cost

def beam_search_prune(
    paths: List[List[Tuple[float, float]]],
    cones: np.ndarray,
    colors: np.ndarray,
    beam_width: int
) -> List[List[Tuple[float, float]]]:
    """
    Select top-k lowest cost paths from path tree.
    Returns: List of beam_width best paths, sorted by cost (lowest first)
    """
    if len(paths) == 0:
        return []

    # evaluate cost for each path
    path_costs = []
    for path in paths:
        cost = evaluate_path_cost(np.array(path), cones, colors)
        path_costs.append((path, cost))

    # sort by cost (lowest = best)
    path_costs.sort(key=lambda x: x[1])

    # keep top beam_width paths
    best_paths = [path for path, cost in path_costs[:beam_width]]

    return best_paths

