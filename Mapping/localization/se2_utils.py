import polars as pl

from localization.se2_utils import se2_from_xytheta, transform_points, invert_se2

df = pl.read_csv("data/Rectangle2x.csv")
print(df.columns)
print(df.head()) #Return the first 5 rows of the dataframe


#2D map:
imu_df = df.select([
        pl.col("General.Time (millis)").alias("time_ms"),
        pl.col("ISM330.Accel X (milli-g)").alias("ax_mg"),
        pl.col("ISM330.Accel Y (milli-g)").alias("ay_mg"),
        pl.col("ISM330.Gyro Z (milli-dps)").alias("gyro_z_mdps"),
    ])

print(imu_df)