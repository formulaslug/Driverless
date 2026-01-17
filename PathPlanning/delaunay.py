from scipy.spatial import Delaunay
from collections import defaultdict
import numpy as np
import config as cfg
from numpy.typing import ArrayLike
from typing import Tuple, Dict, Set

def get_midpoints(cones: ArrayLike, colors) -> Tuple[np.ndarray, Dict[Tuple[float, float], Set[Tuple[float, float]]], Delaunay]:
    """
    Generate waypoint graph from cone positions using Delaunay triangulation.

    Returns waypoints (midpoints of triangle edges), adjacency graph, and triangulation.
    """
    points = np.array(cones)    
    tri = Delaunay(points)
    waypoint_graph = defaultdict(set)
    
    idx1 = tri.simplices[:, 0]  # shape: (n_triangles,)
    idx2 = tri.simplices[:, 1]
    idx3 = tri.simplices[:, 2]

    # get coordinates for each vertex (what you already have)
    p1 = points[idx1]  # shape: (n_triangles, 2)
    p2 = points[idx2]
    p3 = points[idx3]

    # get colors for each vertex using the same indices
    colors1 = colors[idx1]  # shape: (n_triangles, 4)
    colors2 = colors[idx2]
    colors3 = colors[idx3]

    # convert color probabilities to color labels
    # np.argmax finds which color has highest probability
    color_label1 = np.argmax(colors1, axis=1)  # shape: (n_triangles,)
    color_label2 = np.argmax(colors2, axis=1)
    color_label3 = np.argmax(colors3, axis=1)

    # create boolean masks for valid edges

    valid_p1p2 = (
        ((color_label1 == cfg.CONE_COLOR_BLUE) & (color_label2 == cfg.CONE_COLOR_YELLOW)) |
        ((color_label1 == cfg.CONE_COLOR_YELLOW) & (color_label2 == cfg.CONE_COLOR_BLUE))
    )  # shape: (n_triangles,) boolean array

    valid_p1p3 = (
        ((color_label1 == cfg.CONE_COLOR_BLUE) & (color_label3 == cfg.CONE_COLOR_YELLOW)) |
        ((color_label1 == cfg.CONE_COLOR_YELLOW) & (color_label3 == cfg.CONE_COLOR_BLUE))
    )

    valid_p2p3 = (
        ((color_label2 == cfg.CONE_COLOR_BLUE) & (color_label3 == cfg.CONE_COLOR_YELLOW)) |
        ((color_label2 == cfg.CONE_COLOR_YELLOW) & (color_label3 == cfg.CONE_COLOR_BLUE))
    )
    # Compute midpoints of each edge and store
    wayp1p2 = ((p1 + p2) / 2)
    wayp1p3 = ((p1 + p3) / 2)
    wayp2p3 = ((p2 + p3) / 2)
    
    valid_wayp1p2 = wayp1p2[valid_p1p2]  # only rows where mask is True
    valid_wayp1p3 = wayp1p3[valid_p1p3]
    valid_wayp2p3 = wayp2p3[valid_p2p3]
    
    # build graph with only valid waypoints
    # iterate over all triangles and add valid edges from each
    for i in range(len(tri.simplices)):
        waypoints_in_triangle = []

        if valid_p1p2[i]:
            wp = tuple(np.round(wayp1p2[i], decimals=6))
            waypoints_in_triangle.append(wp)

        if valid_p1p3[i]:
            wp = tuple(np.round(wayp1p3[i], decimals=6))
            waypoints_in_triangle.append(wp)

        if valid_p2p3[i]:
            wp = tuple(np.round(wayp2p3[i], decimals=6))
            waypoints_in_triangle.append(wp)

        # connect all valid waypoints within same triangle
        for j, wp1 in enumerate(waypoints_in_triangle):
            for wp2 in waypoints_in_triangle[j+1:]:
                waypoint_graph[wp1].add(wp2)
                waypoint_graph[wp2].add(wp1)

    # stack all the waypoint arrays into one vertically
    waypoints = np.vstack([valid_wayp1p2, valid_wayp1p3, valid_wayp2p3])
    waypoints = np.unique(waypoints, axis=0)
        
    return waypoints, waypoint_graph, tri
