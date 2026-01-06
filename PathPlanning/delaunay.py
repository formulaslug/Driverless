from scipy.spatial import Delaunay
from collections import defaultdict
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, Dict, Set

def get_midpoints(cones: ArrayLike) -> Tuple[np.ndarray, Dict[Tuple[float, float], Set[Tuple[float, float]]], Delaunay]:
    """
    Generate waypoint graph from cone positions using Delaunay triangulation.

    Returns waypoints (midpoints of triangle edges), adjacency graph, and triangulation.
    """
    points = np.array(cones)    
    tri = Delaunay(points)
    waypoint_graph = defaultdict(set)
    
    # Vectorize this for speedup instead of using a loop
    # Get coordinates of each point
    p1 = points[tri.simplices[:,0]]
    p2 = points[tri.simplices[:,1]]
    p3 = points[tri.simplices[:,2]]
    
    # Compute midpoints of each edge and store
    wayp1p2 = ((p1 + p2) / 2)
    wayp1p3 = ((p1 + p3) / 2)
    wayp2p3 = ((p2 + p3) / 2)
    
    for i in range(len(wayp1p2)):
        wp1, wp2, wp3 = wayp1p2[i], wayp1p3[i], wayp2p3[i]
        # Convert to tuples (hashable for dict keys)
        wp1_tuple = tuple(np.round(wp1, decimals=6))
        wp2_tuple = tuple(np.round(wp2, decimals=6))
        wp3_tuple = tuple(np.round(wp3, decimals=6))
        # Each waypoint connects to the other 2 in the triangle
        waypoint_graph[wp1_tuple].add(wp2_tuple)
        waypoint_graph[wp1_tuple].add(wp3_tuple)
        waypoint_graph[wp2_tuple].add(wp1_tuple)
        waypoint_graph[wp2_tuple].add(wp3_tuple)
        waypoint_graph[wp3_tuple].add(wp1_tuple)
        waypoint_graph[wp3_tuple].add(wp2_tuple)
      
    # Stack all the waypoint arrays into one vertically
    waypoints = np.unique(np.vstack([wayp1p2, wayp1p3, wayp2p3]), axis=0)
        
    return waypoints, waypoint_graph, tri
