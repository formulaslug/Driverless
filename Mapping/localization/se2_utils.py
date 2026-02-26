import math
import numpy as np
import polars as pl

# Pose & transformation ulitites

def se2_from_xytheta(x: float, y: float, theta: float) -> np.ndarray:

    sin_t = math.sin(theta) #The sin of the rotation angle theta
    cos_t = math.cos(theta) #The cos of the rotation angle theta

    T = np.array([  [cos_t, -sin_t, x],
                    [sin_t,  cos_t, y],
                    [0,  0, 1]], dtype=float) 
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray: 
 
    assert T.shape == (3, 3)
    assert pts.ndim == 2 and pts.shape[1] == 2 
    N = pts.shape[0] 
    
    homog = np.hstack([pts, np.ones((N, 1))]) 
    transformed = (T @ homog.T).T 
    return transformed[0:, 0:2]


def invert_se2(T: np.ndarray) -> np.ndarray:
    assert T.shape == (3, 3) 
    R = T[0:2, 0:2] 
    t = T[0:2, 2] 
    R_inv = R.T 
    t_inv = -R_inv @ t 
    T_inv = np.eye(3) 
    T_inv[0:2, 0:2] = R_inv 
    T_inv[0:2, 2] = t_inv
    return T_inv