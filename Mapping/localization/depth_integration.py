"""
Depth Estimation Integration Module

Input Expected from Depth Team:
    - 2D array of floats (H x W)
    - Each pixel value = distance in meters
    - Matches camera image dimensions exactly

Output:
    - Function to query depth at any pixel location
"""

import numpy as np

class DepthMap:
    """
    Wrapper for depth estimation data.
    Provides easy access to depth at pixel locations.
    """
    
    def __init__(self, depth_array: np.ndarray):
        """
        Initialize depth map.
        
        Args:
            depth_array: 2D numpy array (H x W) of distances in meters
        """
        if depth_array.ndim != 2:
            raise ValueError(f"Depth array must be 2D, got shape {depth_array.shape}")
        
        self.depth_array = depth_array
        self.height = depth_array.shape[0]
        self.width = depth_array.shape[1]
    
    def get_depth_at_pixel(self, pixel_u: int, pixel_v: int) -> float:
        """
        Get depth (distance) at a specific pixel location.
        
        Args:
            pixel_u: Horizontal pixel coordinate (column)
            pixel_v: Vertical pixel coordinate (row)
            
        Returns:
            Distance in meters at that pixel
        """
        # Check bounds
        if pixel_v < 0 or pixel_v >= self.height:
            raise ValueError(f"pixel_v={pixel_v} out of bounds [0, {self.height})")
        if pixel_u < 0 or pixel_u >= self.width:
            raise ValueError(f"pixel_u={pixel_u} out of bounds [0, {self.width})")
        
        # Return depth at this pixel
        return float(self.depth_array[pixel_v, pixel_u])
    
    def get_depths_batch(self, pixel_u_array: np.ndarray, pixel_v_array: np.ndarray) -> np.ndarray:
        """
        Get depths for multiple pixels at once.
        
        Args:
            pixel_u_array: Array of u coordinates
            pixel_v_array: Array of v coordinates
            
        Returns:
            Array of depths in meters
        """
        return self.depth_array[pixel_v_array, pixel_u_array]


def load_depth_map(depth_data) -> DepthMap:
    """
    Load depth map from Depth team's output.
    
    Args:
        depth_data: Either:
            - np.ndarray (H x W) of floats
            - Path to .npy file
            - Other format (to be determined by Depth team)
            
    Returns:
        DepthMap object
    """
    # If it's already a numpy array, use it directly
    if isinstance(depth_data, np.ndarray):
        return DepthMap(depth_data)
    
    # If it's a file path, load it
    if isinstance(depth_data, str):
        if depth_data.endswith('.npy'):
            arr = np.load(depth_data)
            return DepthMap(arr)
        else:
            raise ValueError(f"Unknown depth file format: {depth_data}")
    
    raise TypeError(f"Cannot load depth map from type: {type(depth_data)}")