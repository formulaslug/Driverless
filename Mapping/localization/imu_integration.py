"""
IMU Data Integration

INPUT: CSV file with IMU data (time, accel, gyro)
OUTPUT: Vehicle pose estimates (x, y, theta) over time

This module loads and processes IMU data to estimate vehicle trajectory.
"""

import polars as pl


def load_imu_data(csv_path: str) -> pl.DataFrame:
    """
    Load and preprocess IMU data from CSV.
    
    INPUT:
        csv_path: Path to IMU CSV file
        Expected columns:
            - "General.Time (millis)"
            - "ISM330.Accel X (milli-g)"
            - "ISM330.Accel Y (milli-g)"
            - "ISM330.Gyro Z (milli-dps)"
    
    OUTPUT:
        DataFrame with columns:
            - time_ms: Time in milliseconds
            - ax_mg, ay_mg: Acceleration in milli-g
            - gyro_z_mdps: Gyro Z in milli-degrees/second
    """
    df = pl.read_csv(csv_path)
    
    # Select and rename relevant columns
    imu_df = df.select([
        pl.col("General.Time (millis)").alias("time_ms"),
        pl.col("ISM330.Accel X (milli-g)").alias("ax_mg"),
        pl.col("ISM330.Accel Y (milli-g)").alias("ay_mg"),
        pl.col("ISM330.Gyro Z (milli-dps)").alias("gyro_z_mdps"),
    ])
    
    return imu_df


def estimate_pose_from_imu(imu_df: pl.DataFrame, v_forward: float = 3.0) -> pl.DataFrame:
    """
    Estimate vehicle pose (x, y, theta) from IMU data using dead reckoning.
    
    INPUT:
        imu_df: DataFrame from load_imu_data()
        v_forward: Assumed constant forward velocity (m/s)
    
    OUTPUT:
        DataFrame with columns:
            - time_s: Time in seconds
            - x: Position forward (meters)
            - y: Position left (meters)
            - theta: Heading angle (radians)
    
    NOTE: This is simple dead reckoning. It will drift over time.
          Later, replace with Kimera-VIO for better accuracy.
    """
    # Sort by time and convert to seconds
    imu_df = imu_df.sort("time_ms").with_columns(
        (pl.col("time_ms") / 1000.0).alias("time_s")
    )
    
    # Convert gyro from milli-deg/s to rad/s
    # Formula: rad/s = (deg/s) * (π/180) = (milli-deg/s) / 1000 * (π/180)
    #        = (milli-deg/s) / 57295.8
    imu_df = imu_df.with_columns(
        (pl.col("gyro_z_mdps") / 57295.8).alias("gyro_z_rads")
    )
    
    # Calculate time difference between samples
    imu_df = imu_df.with_columns(
        pl.col("time_s").diff().fill_null(0.0).alias("dt")
    )
    
    # Integrate gyro to get heading change
    imu_df = imu_df.with_columns(
        (pl.col("gyro_z_rads") * pl.col("dt")).alias("dtheta")
    ).with_columns(
        pl.col("dtheta").cum_sum().alias("theta")  # Cumulative heading
    )
    
    # Calculate distance traveled forward
    imu_df = imu_df.with_columns(
        (v_forward * pl.col("dt")).alias("ds")
    )
    
    # Project forward motion into world coordinates
    imu_df = imu_df.with_columns(
        (pl.col("ds") * pl.col("theta").cos()).alias("dx_w"),  # X increment
        (pl.col("ds") * pl.col("theta").sin()).alias("dy_w"),  # Y increment
    )
    
    # Integrate to get absolute position
    imu_df = imu_df.with_columns(
        pl.col("dx_w").cum_sum().alias("x"),  # Total X position
        pl.col("dy_w").cum_sum().alias("y"),  # Total Y position
    )
    
    # Return just the pose information
    return imu_df.select(["time_s", "x", "y", "theta"])


# Example usage
if __name__ == "__main__":
    print("Testing IMU Integration.")
    
    # Load IMU data
    imu_data = load_imu_data("data/Rectangle2x.csv")
    print(f"Loaded {len(imu_data)} IMU samples")
    print(imu_data.head())
    
    # Estimate poses
    poses = estimate_pose_from_imu(imu_data, v_forward=3.0)
    print(f"\nEstimated {len(poses)} poses")
    print(poses.head())