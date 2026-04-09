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

        self.x_camera = x_camera
        self.y_camera = y_camera
        self.z_camera = z_camera

        if camera_intrinsics is None:
            raise ValueError("Must provide camera_intrinsics dict")

        for key in ("fx", "fy", "cx", "cy"):
            if key not in camera_intrinsics:
                raise ValueError(f"camera_intrinsics missing required key '{key}'")

        self.fx = camera_intrinsics['fx']
        self.fy = camera_intrinsics['fy']
        self.cx = camera_intrinsics['cx']
        self.cy = camera_intrinsics['cy']

        if self.fx == 0 or self.fy == 0:
            raise ValueError("camera_intrinsics fx and fy must be non-zero")

    def pixel_to_camera_frame(self, pixel_u: int, pixel_v: int, depth: float) -> tuple:
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")

        x_cam = (pixel_u - self.cx) * depth / self.fx
        y_cam = (pixel_v - self.cy) * depth / self.fy
        z_cam = depth

        return x_cam, y_cam, z_cam

    def camera_to_vehicle_frame(self, x_cam: float, y_cam: float, z_cam: float) -> tuple:
        x_veh = z_cam + self.x_camera
        y_veh = -x_cam + self.y_camera
        z_veh = -y_cam + self.z_camera

        return x_veh, y_veh, z_veh

    def transform_detections(self, detections_df: pl.DataFrame) -> pl.DataFrame:
        for col in ("pixel_u", "pixel_v", "depth_m"):
            if col not in detections_df.columns:
                raise ValueError(f"detections_df missing required column '{col}'")

        vehicle_coords = []

        for row in detections_df.iter_rows(named=True):
            x_cam, y_cam, z_cam = self.pixel_to_camera_frame(
                row['pixel_u'],
                row['pixel_v'],
                row['depth_m']
            )

            x_veh, y_veh, z_veh = self.camera_to_vehicle_frame(x_cam, y_cam, z_cam)

            vehicle_coords.append({
                'x_vehicle': x_veh,
                'y_vehicle': y_veh,
                'z_vehicle': z_veh
            })

        result = detections_df.hstack(pl.DataFrame(vehicle_coords))

        for col in ("x_vehicle", "y_vehicle", "z_vehicle"):
            if col not in result.columns:
                raise ValueError(f"transform_detections failed to produce column '{col}'")

        return result

    def info(self):
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


RPI_CAM_INTRINSICS = {
    'fx': 1964,
    'fy': 1964,
    'cx': 2304.0,
    'cy': 1296.0
}

camera_transform = CameraToVehicleTransform(
    x_camera=0.0,
    y_camera=0.0,
    z_camera=0.0,
    camera_intrinsics=RPI_CAM_INTRINSICS
)

if __name__ == "__main__":
    camera_transform.info()