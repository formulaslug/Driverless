import polars as pl

def load_imu_data(csv_path: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path)
    
    imu_df = df.select([
        pl.col("General.Time (millis)").alias("time_ms"),
        pl.col("ISM330.Accel X (milli-g)").alias("ax_mg"),
        pl.col("ISM330.Accel Y (milli-g)").alias("ay_mg"),
        pl.col("ISM330.Gyro Z (milli-dps)").alias("gyro_z_mdps"),
    ])
    
    return imu_df


def estimate_pose_from_imu(imu_df: pl.DataFrame) -> pl.DataFrame:
    """
    Integration chain:
        ax_mg --> a_forward --> v_forward --> ds --> (dx_w, dy_w) --> (x, y)
        gyro_z_mdps --> gyro_z_rads --> dtheta --> theta
    """
    
    imu_df = imu_df.sort("time_ms").with_columns(
        (pl.col("time_ms") / 1000.0).alias("time_s")
    )
    
    imu_df = imu_df.with_columns(
        pl.col("time_s").diff().fill_null(0.0).alias("dt")
    )
    
    imu_df = imu_df.with_columns(
        (pl.col("ax_mg") / 1000.0 * 9.81).alias("a_forward")
    )
    
    imu_df = imu_df.with_columns(
        (pl.col("a_forward") * pl.col("dt")).alias("dv")
    )
    

    imu_df = imu_df.with_columns(
        pl.col("dv").cum_sum().alias("v_forward")
    )
    

    imu_df = imu_df.with_columns(
        (pl.col("v_forward") * pl.col("dt")).alias("ds")
    )
 
    imu_df = imu_df.with_columns(
        (pl.col("gyro_z_mdps") / 57295.8).alias("gyro_z_rads")
    )
    

    imu_df = imu_df.with_columns(
        (pl.col("gyro_z_rads") * pl.col("dt")).alias("dtheta")
    )
    
    imu_df = imu_df.with_columns(
        pl.col("dtheta").cum_sum().alias("theta")
    )


    return imu_df.select([
        "time_s",     # Timestamp (seconds)
        "dt",         # Time increment (seconds)
        "a_forward",  # Forward acceleration (m/s^2)
        "v_forward",  # Forward velocity (m/s) integrated
        "ds",         # Distance increment (meters) how much the car moved?
        "dtheta",     # Heading change increment (radians) how much the car rotated?
        "theta",      # Cumulative heading (radians)
    ])