import math
import numpy as np
import polars as pl

#Pose & transformation ulitites

def se2_from_xytheta(x: float, y: float, theta: float) -> np.ndarray:
    """Create and returns a 2D SE(2) transformation matrix from x, y, and theta."""
    """SE(2) is the special Euclidean group of 2-D transformations, combines a pure rotation (SO(2)) w/ a translation (xy-vector)."""
    
    """c & s are used to build the 2x2 rotation matrix inside the 3x3 homogeneous SE(2) transformation matrix."""
    cos_t = math.cos(theta) #The cos of the rotation angle theta
    sin_t = math.sin(theta) #The sin of the rotation angle theta
    
    """Construct the SE(2) transformation matrix
    The bottom row [0, 0, 1] makes it a homogeneous transformation matrix.
    A homogeneous transformation matrix allows for the representation of both rotation 
    and translation in a single matrix, enabling easy composition of multiple transformations."""
    T = np.array([  [cos_t, -sin_t, x],
                    [sin_t,  cos_t, y],
                    [0,  0, 1]], dtype=float) 
    return T

"""#Defines a func that takes a 3x3 transformation matrix T (SE(2)) and 
    an array of (N x 2) 2D points"""
def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray: 
    """
    Apply SE(2) transform T (3x3) to 2D points pts (N x 2).
    Returns transformed points (N x 2).
    """
    assert T.shape == (3, 3) # Makes sure T is a 3x3, the size for a 2D homogeneous transformation matrix
    assert pts.ndim == 2 and pts.shape[1] == 2 #Verifies input points are a two-dimensional array, with each row a point
    N = pts.shape[0] #Counts how many points you're transforming, cannot be more than a 3x3 matrix. 
    
    """(N x 3) Adds a 1 to each point to make it [x, y, 1] ("homogeneous coordinates") 
    allows both rotation and translation with one matrix multiply
    hstack = horizontal stack
    pts has shape (N, 2) → [x, y] for each point
    np.ones((N, 1)) is a column of ones with shape (N, 1).
    hstack joins them to make shape (N, 3) → [x, y, 1]."""
    homog = np.hstack([pts, np.ones((N, 1))]) 

    """"@ is the multiplication matrix, same as np.matmul(T, homog)
    T is a 3x3 transform and homog.T is a 3xN matrix (bc we transpose)."""
    
    """Multiply every point [x, y, 1] by the transform T to rotate + translate it.
    The .T at the end just flips it back to shape (N, 3)."""
    transformed = (T @ homog.T).T #(N x 3) Multiplies all the (x, y, 1) points by T, rotating and translating every point
    return transformed[0:, 0:2] #Returns just the new (x, y) columns, discarding the homogeneous "1".


"""This function takes a NumPy array T (2D SE(2)) matrix
   and returns its inverse as a NumPy array."""
def invert_se2(T: np.ndarray) -> np.ndarray:
    """Invert a 2D SE(2) transform. Maybe useful to go global -> local (vehicle frame)."""
    assert T.shape == (3, 3) #Ensure T is a 3x3 matrix
    R = T[0:2, 0:2] #Extract the 2x2 rotation part from the top-left of T
    t = T[0:2, 2] #Extract the 2x1 translation vector from the top-right of T
    R_inv = R.T #Inverse of rotation matrix is its transpose (rows become columns and columns become rows)
    t_inv = -R_inv @ t #Inverse translation is -R^T * t, found it from Articulated Robotics.xyz
    T_inv = np.eye(3) #Create a 3x3 identity matrix to hold the inverse transform
    T_inv[0:2, 0:2] = R_inv #Set the top-left 2x2 part to the inverse rotation
    T_inv[0:2, 2] = t_inv #Set the top-right 2x1 part to the inverse translation
    return T_inv