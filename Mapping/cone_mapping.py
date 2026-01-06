"""
NOTE:
This file is meant to be imported, not run directly.
Run pose_playground.py from the project root instead.
"""

import polars as pl
import numpy as np
from localization.se2_utils import se2_from_xytheta, transform_points

def cones_vehicle_to_global(
    pose_df: pl.DataFrame,
    cones_vehicle: np.ndarray,
    sample_every: int = 1,
) -> pl.DataFrame:
    rows = []
    sample = pose_df[::sample_every]
    for pose_id, row in enumerate(sample.iter_rows(named=True)):
        T = se2_from_xytheta(row["x"], row["y"], row["theta"])
        pts_global = transform_points(T, cones_vehicle)
        for cone_id, (xg, yg) in enumerate(pts_global):
            rows.append({
                "pose_id": pose_id,
                "cone_id": cone_id,
                "x_global": float(xg),
                "y_global": float(yg),
            })
    return pl.DataFrame(rows)

#Filtering Cone Observations
def filter_cone_outliers(cones_df: pl.DataFrame) -> pl.DataFrame:
    """
    First-pass cone filtering to remove obvious outliers.
    Assumes:
      x_v_m: forward distance in meters (vehicle frame)
      y_v_m: lateral distance in meters (vehicle frame, left positive)
    """
    return (
        cones_df
        .filter(pl.col("x_v_m").is_finite() & pl.col("y_v_m").is_finite())
        .filter(pl.col("x_v_m").abs() < 60.0) #Cone is more than 60 meters ahead or behind is removed
        .filter(pl.col("y_v_m").abs() < 20.0) #Cone is more than 20 meters to the left or right is removed
    )

def bin_and_average_cones(cones_df: pl.DataFrame) -> pl.DataFrame:
    """
    Collapse nearby cone detections (per pose/frame) into one averaged cone.
    Assumes:
      - pose_id: which pose/frame this detection belongs to
      - x_v_m, y_v_m: vehicle-frame coordinates in meters
    """
    df = (
        cones_df
        .with_columns([
            (pl.col("x_v_m") * 2).round().alias("x_bin"),  #0.5 m bins
            (pl.col("y_v_m") * 2).round().alias("y_bin"),
        ])
        .group_by(["pose_id", "x_bin", "y_bin"])
        .agg([
            pl.mean("x_v_m").alias("x_v_m_avg"),
            pl.mean("y_v_m").alias("y_v_m_avg"),
            pl.len().alias("hit_count"),
        ])
    )
    return df

##Old code for cones_vehicle_to_global before refactor to function

# #This list will store all cone observations
# #Each entry corresponds to one cone seen at one pose
# rows = []

# #Subsample poses so we don't spam output
# K = 20
# sample = pose_df[::K]

# #row is a dictionary that has keys time_s, x, y, theta
# #pose_id = index of the pose (time step)
# for pose_id, row in enumerate(sample.iter_rows(named=True)):
#     #SE(2): vehicle frame -> world frame for this pose
#     T_world_vehicle = se2_from_xytheta(row["x"], row["y"], row["theta"])

#     #Convert relative cone positions into global frame
#     pts_global = transform_points(T_world_vehicle, pts_vehicle)

#     #Loop over each cone seen at this pose
#     for cone_id, (xg, yg) in enumerate(pts_global):
#         rows.append({
#             "pose_id": pose_id, #Which time step
#             "cone_id": cone_id, #Which cone at that time
#             "x_global": float(xg),
#             "y_global": float(yg),
#         })

# global_cones_df = pl.DataFrame(rows)
# print("First few global cone positions:")
# print(global_cones_df.head())