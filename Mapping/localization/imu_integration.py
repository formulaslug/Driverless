import polars as pl

def load_imu_data(csv_path: str) -> pl.DataFrame:
    if not csv_path or not isinstance(csv_path, str):
        raise ValueError("csv_path must be a non-empty string")

    df = pl.read_csv(csv_path)

    required_cols = [
        "General.Time (millis)",
        "ISM330.Accel X (milli-g)",
        "ISM330.Accel Y (milli-g)",
        "ISM330.Gyro Z (milli-dps)",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    imu_df = df.select([
        pl.col("General.Time (millis)").alias("time_ms"),
        pl.col("ISM330.Accel X (milli-g)").alias("ax_mg"),
        pl.col("ISM330.Accel Y (milli-g)").alias("ay_mg"),
        pl.col("ISM330.Gyro Z (milli-dps)").alias("gyro_z_mdps"),
    ])

    if imu_df.is_empty():
        raise ValueError("IMU CSV loaded but contains no rows")

    return imu_df

def estimate_pose_from_imu(imu_df: pl.DataFrame) -> pl.DataFrame:
    """
    Integration chain:
        ax_mg --> a_forward --> v_forward --> ds --> (dx_w, dy_w) --> (x, y)
        gyro_z_mdps --> gyro_z_rads --> dtheta --> theta
    """
    required_cols = ["time_ms", "ax_mg", "ay_mg", "gyro_z_mdps"]
    missing = [c for c in required_cols if c not in imu_df.columns]
    if missing:
        raise ValueError(f"imu_df missing required columns: {missing}")
    if imu_df.is_empty():
        raise ValueError("imu_df is empty")

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

    result = imu_df.select([
        "time_s",
        "dt",
        "a_forward",
        "v_forward",
        "ds",
        "dtheta",
        "theta",
    ])

    if result.is_empty():
        raise ValueError("estimate_pose_from_imu produced an empty DataFrame")

    return result