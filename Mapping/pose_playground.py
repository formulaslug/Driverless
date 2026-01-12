import polars as pl
import numpy as np

from localization.se2_utils import se2_from_xytheta
from localization.cone_mapping import cones_vehicle_to_global, filter_cone_outliers
from localization.se3_utlis import se3_from_xyz_rpy, transform_points_se3
from explore_imu import imu_df  # IMU subset

v_forward = 3.0  # m/s, temporary constant forward velocity

# 1) sort and add time in seconds
imu_df = imu_df.sort("time_ms").with_columns(
    (pl.col("time_ms") / 1000).alias("time_s")
)

# 2) convert gyro to rad/s (milli-deg/s -> rad/s)
imu_df = imu_df.with_columns(
    (pl.col("gyro_z_mdps") / 57295.8).alias("gyro_z_rads")
)

# 3) delta time
imu_df = imu_df.with_columns(
    (pl.col("time_s").diff().fill_null(0.0)).alias("dt")
)

# 4) integrate heading
imu_df = imu_df.with_columns(
    (pl.col("gyro_z_rads") * pl.col("dt")).alias("dtheta")
).with_columns(
    pl.col("dtheta").cum_sum().alias("theta")
)

# 5) forward distance each step
imu_df = imu_df.with_columns(
    (v_forward * pl.col("dt")).alias("ds")
)

# 6) project into world frame, integrate
imu_df = imu_df.with_columns(
    (pl.col("ds") * pl.col("theta").cos()).alias("dx_w"),
    (pl.col("ds") * pl.col("theta").sin()).alias("dy_w"),
).with_columns(
    pl.col("dx_w").cum_sum().alias("x"),
    pl.col("dy_w").cum_sum().alias("y"),
)

pose_df = imu_df.select(["time_s", "x", "y", "theta"])

if __name__ == "__main__":
    print(pose_df.head())

# Fake cones in vehicle frame (relative map)
pts_vehicle = np.array([
    [5.0,  0.0],
    [6.0,  2.0],
    [6.0, -2.0],
])

N = pts_vehicle.shape[0]
cones_vehicle_df = pl.DataFrame({
    "cone_id": list(range(N)),
    "x_v_m": pts_vehicle[:, 0],
    "y_v_m": pts_vehicle[:, 1],
})

cones_vehicle_df = filter_cone_outliers(cones_vehicle_df)
pts_vehicle_filtered = cones_vehicle_df.select(["x_v_m", "y_v_m"]).to_numpy()

global_cones_df = cones_vehicle_to_global(pose_df, pts_vehicle_filtered, sample_every=20)

if __name__ == "__main__":
    print("First few global cone positions:")
    print(global_cones_df.head())


# --- Camera -> Vehicle example (for now, just a test) ---

# Example intrinsics (replace with calibrated values from OpenCV later)
fx, fy = 600.0, 600.0
cx, cy = 320.0, 240.0

def pixel_depth_to_camera(u, v, depth_m):
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z])

# Example camera pose wrt vehicle: 0.5m forward, 0m left, 0.3m up, no rotation
T_vehicle_camera = se3_from_xyz_rpy(0.5, 0.0, 0.3, 0.0, 0.0, 0.0)

if __name__ == "__main__":
    # Existing debug prints
    print("First few poses:")
    print(pose_df.head())

    # Camera back-projection test
    u, v = 300, 200
    depth_m = 8.0
    P_cam = pixel_depth_to_camera(u, v, depth_m).reshape(1, 3)
    P_veh = transform_points_se3(T_vehicle_camera, P_cam)[0]
    print("Camera-frame point:", P_cam)
    print("Vehicle-frame point:", P_veh, "(use x,y for 2D map)")


# """
# This script performs simple dead-reckoning localization using IMU data.

# Pipeline:
# 1. Load IMU data (time + gyro)
# 2. Integrate gyro Z to get heading (theta)
# 3. Assume constant forward velocity
# 4. Integrate motion to get (x, y) position in the world frame

# Output:
# - x, y, theta over time (robot pose estimate)
# """

# import polars as pl
# import numpy as np

# from localization.se2_utils import se2_from_xytheta, transform_points, invert_se2
# from localization.cone_mapping import (
#     cones_vehicle_to_global,
#     filter_cone_outliers,
#     bin_and_average_cones,
# )
# from explore_imu import imu_df

# #Constants
# #This is a placeholder, real localization would estimate velocity
# v_forward = 3.0  #m/s constant forward velocity

# #Converts millis into seconds
# imu_df = imu_df.sort("time_ms").with_columns( #Sort by time from smallest to largest 
#     (pl.col("time_ms") / 1000).alias("time_s") 
# )

# #Convert milli-deg/s to rad/s 
# #Gyro measures angular velocity, NOT angle
# imu_df = imu_df.with_columns(
#     (pl.col("gyro_z_mdps") / 57295.8).alias("gyro_z_rads") 
# )

# #Calculate time difference between each row in seconds
# #dt represents how much time passed since the previous row
# #In Donkey terms, "How much time passed?"
# imu_df = imu_df.with_columns(
#     (pl.col("time_s").diff().fill_null(0.0)).alias("dt") 
# )

# #Calculate change in heading angle (radians) for each time step
# #In Donkey terms, “How much did I turn in that time?” -> dtheta
# #dtheta = omega * dt
# imu_df = imu_df.with_columns(
#     (pl.col("gyro_z_rads") * pl.col("dt")).alias("dtheta"), 
# )

# #Cumulative sum of change in heading to get absolute heading angle at each time
# #Integrate heading over time to get absolute yaw (theta)
# #In Donkey terms, “Which way am I facing now?” -> theta
# imu_df = imu_df.with_columns(
#     pl.col("dtheta").cum_sum().alias("theta")
# )

# #figure what this does
# imu_df = imu_df.with_columns(
#     (v_forward * pl.col("dt")).alias("ds") #Distance moved along vehicle x
# )

# #Convert forward motion (vehicle frame) into world-frame motion
# #ds is distance traveled along the vehicle's x-axis
# imu_df = imu_df.with_columns(
#     #Project forward motion into world coordinates using current heading
#     (pl.col("ds") * pl.col("theta").cos()).alias("dx_w"), #World x increment
#     (pl.col("ds") * pl.col("theta").sin()).alias("dy_w"), #World y increment
# )

# #Integrate world-frame motion to get absolute position
# #x[i], y[i] represent the estimated global position of the vehicle
# imu_df = imu_df.with_columns(
#     pl.col("dx_w").cum_sum().alias("x"),
#     pl.col("dy_w").cum_sum().alias("y"),
# )

# """
# At this point, imu_df contains the estimated pose over time:
# pose(t) = (x(t), y(t), theta(t))
# """

# #Build a simpler pose_df from imu_df
# pose_df = imu_df.select(["time_s", "x", "y", "theta"]) #x and y are in the world frame?
# print("First few poses:")
# print(pose_df.head())

# pts_vehicle = np.array([
#     [5.0,  0.0],
#     [6.0,  2.0],
#     [6.0, -2.0],
# ])

# N = pts_vehicle.shape[0]
# cones_vehicle_df = pl.DataFrame({
#     "cone_id": list(range(N)), #0..N-1
#     "x_v_m": pts_vehicle[:, 0],
#     "y_v_m": pts_vehicle[:, 1],
# })


# #Apply first-pass filtering (does nothing yet for these fake cones, but ready for real data)
# cones_vehicle_df = filter_cone_outliers(cones_vehicle_df)

# #For now, convert back to NumPy to reuse existing mapping function
# pts_vehicle_filtered = cones_vehicle_df.select(["x_v_m", "y_v_m"]).to_numpy()

# global_cones_df = cones_vehicle_to_global(pose_df, pts_vehicle_filtered, sample_every=20)
# print("First few global cone positions:")
# print(global_cones_df.head())