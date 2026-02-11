"""
Camera to Vehicle Transformation Module

Transforms pixel coordinates + depth into vehicle-frame coordinates.
Assumes camera is mounted at the vehicle origin with no offset.
"""

import numpy as np
import polars as pl

class CameraToVehicleTransform:
    """
    Convert camera pixel coordinates + depth to vehicle-frame (x, y, z).
    
    Camera frame: +X right, +Y down, +Z forward
    Vehicle frame: +X forward, +Y left, +Z up
    """
    
    def __init__(self, x_camera: float = 0.0, y_camera: float = 0.0, 
                 z_camera: float = 0.0, camera_intrinsics: dict = None):
        """
        Args:
            x_camera: Camera X offset in vehicle frame (meters, forward)
            y_camera: Camera Y offset in vehicle frame (meters, left)
            z_camera: Camera Z offset in vehicle frame (meters, up)
            camera_intrinsics: Dict with 'fx', 'fy', 'cx', 'cy'
        """

        # Camera offsets (defult to 0, the camera is at vehicle origin)
        self.x_camera = x_camera
        self.y_camera = y_camera
        self.z_camera = z_camera
        
        # If they are no provided, an error will be raised. 
        if camera_intrinsics is None:
            raise ValueError("Must provide camera_intrinsics dict")
        
        self.fx = camera_intrinsics['fx']
        self.fy = camera_intrinsics['fy']
        self.cx = camera_intrinsics['cx']
        self.cy = camera_intrinsics['cy']
    
    def pixel_to_camera_frame(self, pixel_u: int, pixel_v: int, depth: float) -> tuple:
        """
        Convert pixel coordinates + depth to 3D point in camera frame.
        
        Args:
            pixel_u: Horizontal pixel coordinate (column)
            pixel_v: Vertical pixel coordinate (row)
            depth: Depth in meters (from depth estimation)
            
        Returns:
            (x_cam, y_cam, z_cam) in camera frame
        """
        # Pinhole camera model
        x_cam = (pixel_u - self.cx) * depth / self.fx # how far right/left from center
        y_cam = (pixel_v - self.cy) * depth / self.fy # how far down from center
        z_cam = depth
        
        return x_cam, y_cam, z_cam
    
    def camera_to_vehicle_frame(self, x_cam: float, y_cam: float, z_cam: float) -> tuple:
        """
        Transform camera frame coordinates to vehicle frame.
        
        Camera: +X right, +Y down, +Z forward
        Vehicle: +X forward, +Y left, +Z up
        
        Args:
            x_cam, y_cam, z_cam: Point in camera frame
            
        Returns:
            (x_veh, y_veh, z_veh) in vehicle frame
        """
        # Rotation: camera Z -> vehicle X, camera -X -> vehicle Y, camera -Y -> vehicle Z
        x_veh = z_cam + self.x_camera
        y_veh = -x_cam + self.y_camera
        z_veh = -y_cam + self.z_camera
        
        return x_veh, y_veh, z_veh
    
    def transform_detections(self, detections_df: pl.DataFrame) -> pl.DataFrame:
        """
        Transform all detections from pixel + depth to vehicle coordinates.
        
        Args:
            detections_df: DataFrame with columns:
                - pixel_u, pixel_v: Pixel coordinates
                - depth_m: Depth in meters
                
        Returns:
            DataFrame with added columns:
                - x_vehicle: Forward distance (meters)
                - y_vehicle: Left distance (meters)
                - z_vehicle: Up distance (meters)
        """
        vehicle_coords = []
        
        for row in detections_df.iter_rows(named=True):
            # Pixel + depth → camera frame
            x_cam, y_cam, z_cam = self.pixel_to_camera_frame(
                row['pixel_u'], 
                row['pixel_v'], 
                row['depth_m']
            )
            
            # Camera frame → vehicle frame
            x_veh, y_veh, z_veh = self.camera_to_vehicle_frame(x_cam, y_cam, z_cam)
            
            vehicle_coords.append({
                'x_vehicle': x_veh,
                'y_vehicle': y_veh,
                'z_vehicle': z_veh
            })
        
        return detections_df.hstack(pl.DataFrame(vehicle_coords))
    
    def info(self):
        """Print camera configuration info."""
        print("--- Camera to Vehicle Transform ---")
        print(f"Camera intrinsics:")
        print(f"  fx={self.fx:.1f}, fy={self.fy:.1f}")
        print(f"  cx={self.cx:.1f}, cy={self.cy:.1f}")
        print(f"\nCamera position in vehicle frame:")
        print(f"  x={self.x_camera:.3f}m (forward)")
        print(f"  y={self.y_camera:.3f}m (left)")
        print(f"  z={self.z_camera:.3f}m (up)")
        print(f"\nFrame convention:")
        print(f"  Camera: +X right, +Y down, +Z forward")
        print(f"  Vehicle: +X forward, +Y left, +Z up")


# DEFAULT CAMERA CONFIGURATION (RPi Cam v3 Wide)

RPI_CAM_INTRINSICS = {
    'fx': 1964,      # from 2.75mm focal length
    'fy': 1964,
    'cx': 2304.0,    # 4608/2
    'cy': 1296.0     # 2592/2
}

# No offset (camera = vehicle origin)
camera_transform = CameraToVehicleTransform(
    x_camera=0.0,
    y_camera=0.0,
    z_camera=0.0,
    camera_intrinsics=RPI_CAM_INTRINSICS
)

# Print config
if __name__ == "__main__":
    camera_transform.info()