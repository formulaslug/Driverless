import polars as pl

_G = 9.81 # m/s²
_MDPS_TO_RADS = 1.0 / 57295.8   # milli-degrees-per-second -> radians/second


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
        ax_mg       -> a_forward -> v_forward -> ds
        gyro_z_mdps -> dtheta   -> theta
    """
    required_cols = ["time_ms", "ax_mg", "ay_mg", "gyro_z_mdps"]
    missing = [c for c in required_cols if c not in imu_df.columns]
    if missing:
        raise ValueError(f"imu_df missing required columns: {missing}")
    if imu_df.is_empty():
        raise ValueError("imu_df is empty")

    # Pass 1 - all expressions that depend only on raw input columns
    # a_forward, dt, and dtheta are all needed by pass 2
    imu_df = (
        imu_df
        .sort("time_ms")
        .with_columns(
            (pl.col("time_ms") / 1000.0).alias("time_s"),
            (pl.col("ax_mg") / 1000.0 * _G).alias("a_forward"),
        )
        .with_columns(
            pl.col("time_s").diff().fill_null(0.0).alias("dt"),
        )
    )

    # Pass 2 - columns that depend on dt (computed in pass 1)
    # v_forward = cumsum(a_forward * dt); ds = v_forward * dt
    # dtheta    = gyro_z_mdps * MDPS_TO_RADS * dt; theta = cumsum(dtheta)
    imu_df = imu_df.with_columns(
        (pl.col("a_forward") * pl.col("dt")).cum_sum().alias("v_forward"),
        (pl.col("gyro_z_mdps") * _MDPS_TO_RADS * pl.col("dt")).alias("dtheta"),
    ).with_columns(
        (pl.col("v_forward") * pl.col("dt")).alias("ds"),
        pl.col("dtheta").cum_sum().alias("theta"),
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