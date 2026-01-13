import numpy as np
from scipy.interpolate import splprep, splev
from typing import Tuple
from config import SPLINE_DEGREE, SPLINE_SMOOTHING_FACTOR

def smooth_path(waypoints: np.ndarray, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth discrete waypoints using cubic spline interpolation.

    Returns smoothed path coordinates and curvature array.
    """
    # u: parameter values of original waypoints
    #    that measure how far along the path each waypoint is
    #    given as a normalized value [0,1]
    # tck: tuple(knots, coefficients, degree of spline)
    tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=SPLINE_SMOOTHING_FACTOR, k=SPLINE_DEGREE)

    # get num_points evenly spaced points
    u_fine = np.linspace(0, 1, num_points)

    # get smoothed x,y coordinates and path
    x_smooth, y_smooth = splev(u_fine, tck)
    
    # stack all the smoothed (x,y) coordinates into one array
    smooth_path = np.column_stack([x_smooth, y_smooth])

    # get curvature (curvature = 1/radius)
    dx_dt, dy_dt = splev(u_fine, tck, der=1)
    d2x_dt2, d2y_dt2 = splev(u_fine, tck, der=2)

    # curvature equation using 1st and 2nd derivative
    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5

    return smooth_path, curvature
