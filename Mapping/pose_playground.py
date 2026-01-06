"""
This script performs simple dead-reckoning localization using IMU data.

Pipeline:
1. Load IMU data (time + gyro)
2. Integrate gyro Z to get heading (theta)
3. Assume constant forward velocity
4. Integrate motion to get (x, y) position in the world frame

Output:
- x, y, theta over time (robot pose estimate)
"""

import polars as pl
import numpy as np

from localization.se2_utils import se2_from_xytheta, transform_points, invert_se2
from localization.cone_mapping import (
    cones_vehicle_to_global,
    filter_cone_outliers,
    bin_and_average_cones,
)
from explore_imu import imu_df

#Constants
#This is a placeholder, real localization would estimate velocity
v_forward = 3.0  #m/s constant forward velocity

#Converts millis into seconds
imu_df = imu_df.sort("time_ms").with_columns( #Sort by time from smallest to largest 
    (pl.col("time_ms") / 1000).alias("time_s") 
)

#Convert milli-deg/s to rad/s 
#Gyro measures angular velocity, NOT angle
imu_df = imu_df.with_columns(
    (pl.col("gyro_z_mdps") / 57295.8).alias("gyro_z_rads") 
)

#Calculate time difference between each row in seconds
#dt represents how much time passed since the previous row
#In Donkey terms, "How much time passed?"
imu_df = imu_df.with_columns(
    (pl.col("time_s").diff().fill_null(0.0)).alias("dt") 
)

#Calculate change in heading angle (radians) for each time step
#In Donkey terms, “How much did I turn in that time?” -> dtheta
#dtheta = omega * dt
imu_df = imu_df.with_columns(
    (pl.col("gyro_z_rads") * pl.col("dt")).alias("dtheta"), 
)

#Cumulative sum of change in heading to get absolute heading angle at each time
#Integrate heading over time to get absolute yaw (theta)
#In Donkey terms, “Which way am I facing now?” -> theta
imu_df = imu_df.with_columns(
    pl.col("dtheta").cum_sum().alias("theta")
)

#figure what this does
imu_df = imu_df.with_columns(
    (v_forward * pl.col("dt")).alias("ds") #Distance moved along vehicle x
)

#Convert forward motion (vehicle frame) into world-frame motion
#ds is distance traveled along the vehicle's x-axis
imu_df = imu_df.with_columns(
    #Project forward motion into world coordinates using current heading
    (pl.col("ds") * pl.col("theta").cos()).alias("dx_w"), #World x increment
    (pl.col("ds") * pl.col("theta").sin()).alias("dy_w"), #World y increment
)

#Integrate world-frame motion to get absolute position
#x[i], y[i] represent the estimated global position of the vehicle
imu_df = imu_df.with_columns(
    pl.col("dx_w").cum_sum().alias("x"),
    pl.col("dy_w").cum_sum().alias("y"),
)

"""
At this point, imu_df contains the estimated pose over time:
pose(t) = (x(t), y(t), theta(t))
"""

#Build a simpler pose_df from imu_df
pose_df = imu_df.select(["time_s", "x", "y", "theta"])
print("First few poses:")
print(pose_df.head())

pts_vehicle = np.array([
    [5.0,  0.0],
    [6.0,  2.0],
    [6.0, -2.0],
])

N = pts_vehicle.shape[0]
cones_vehicle_df = pl.DataFrame({
    "cone_id": list(range(N)), #0..N-1
    "x_v_m": pts_vehicle[:, 0],
    "y_v_m": pts_vehicle[:, 1],
})


#Apply first-pass filtering (does nothing yet for these fake cones, but ready for real data)
cones_vehicle_df = filter_cone_outliers(cones_vehicle_df)

#For now, convert back to NumPy to reuse your existing mapping function
pts_vehicle_filtered = cones_vehicle_df.select(["x_v_m", "y_v_m"]).to_numpy()

global_cones_df = cones_vehicle_to_global(pose_df, pts_vehicle_filtered, sample_every=20)
print("First few global cone positions:")
print(global_cones_df.head())