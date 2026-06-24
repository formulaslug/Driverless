import numpy as np
import polars as pl
from typing import Optional, Tuple, List

from imu_integration import estimate_pose_from_imu
from synchronize import PerceptionSynchronizer
from localization.filter import ConeFilter # Daniel's Filter


class EgoMotionEstimator:

    def __init__(self, imu_df: pl.DataFrame):
        if not isinstance(imu_df, pl.DataFrame) or imu_df.is_empty():
            raise ValueError("imu_df must be a non-empty Polars DataFrame")

        odometry_df = estimate_pose_from_imu(imu_df)

        for col in ("ds", "time_s", "theta"):
            if col not in odometry_df.columns:
                raise ValueError(
                    f"estimate_pose_from_imu output missing column '{col}'"
                )

        self.odometry_df = odometry_df.with_columns(
            pl.col("ds").cum_sum().alias("s_cum")
        )

        # uhhh touches the DataFrame again after __init__
        self.times  = self.odometry_df["time_s"].to_numpy()
        self.s_cum  = self.odometry_df["s_cum"].to_numpy()
        self.theta  = self.odometry_df["theta"].to_numpy()

        if len(self.times) == 0:
            raise ValueError("odometry_df produced empty time array")

    def get_motion_between(self, t_start: float, t_end: float) -> Tuple[float, float, float]:
        if not isinstance(t_start, (int, float)) or not isinstance(t_end, (int, float)):
            raise TypeError(
                f"t_start and t_end must be numeric, got "
                f"{type(t_start)}, {type(t_end)}"
            )
        t_query = np.array([t_start, t_end])

        s_vals     = np.interp(t_query, self.times, self.s_cum)
        theta_vals = np.interp(t_query, self.times, self.theta)

        ds    = s_vals[1] - s_vals[0]
        dyaw  = theta_vals[1] - theta_vals[0]
        theta_start = theta_vals[0]

        dx = ds * np.cos(theta_start)
        dy = ds * np.sin(theta_start)

        return float(dx), float(dy), float(dyaw)


class MappingAndFusion:

    def __init__(self, ego_motion: EgoMotionEstimator, cone_filter: ConeFilter):
        if not isinstance(ego_motion, EgoMotionEstimator):
            raise TypeError("ego_motion must be an EgoMotionEstimator instance")
        if cone_filter is None:
            raise ValueError("cone_filter must not be None")

        self.ego_motion = ego_motion
        self.cone_filter = cone_filter
        self.last_frame_time: Optional[float] = None

    def handle_perception_frame(self, frame: dict) -> None:
        for key in ("time_s", "cone_measurements"):
            if key not in frame:
                raise ValueError(f"perception frame dict missing required key '{key}'")
            
        if not isinstance(frame["cone_measurements"], list):
            raise TypeError("frame['cone_measurements'] must be a list")

        time_s = frame["time_s"]
        cone_measurements = frame["cone_measurements"]

        detections = [
            (x, y, blue, yellow, sOrange, bOrange)
            for (x, y), (blue, yellow, sOrange, bOrange) in cone_measurements
        ]

        if self.last_frame_time is None:
            dx, dy, dyaw = 0.0, 0.0, 0.0
        else:
            dx, dy, dyaw = self.ego_motion.get_motion_between(self.last_frame_time, time_s)

        self.cone_filter.update(detections, dx, dy, dyaw)
        self.last_frame_time = time_s

    def get_cone_map_for_planner(self,) -> List[Tuple[float, float, float, float, float, float]]:
        result = self.cone_filter.getConeMap()

        if not isinstance(result, list):
            raise TypeError(
                f"getConeMap() must return a list, got {type(result)}"
            )

        return result


class PerceptionFusionPipeline:

    def __init__(self, perception_sync: PerceptionSynchronizer, fusion: MappingAndFusion):
        if perception_sync is None:
            raise ValueError("perception_sync must not be None")
        if fusion is None:
            raise ValueError("fusion must not be None")

        self.perception_sync = perception_sync
        self.fusion = fusion

    def on_yolact_result(self, frame_id: int, time_s: float, yolact_output: dict) -> None:
        result = self.perception_sync.add_yolact_result(frame_id, time_s, yolact_output)
        if result is not None:
            self.fusion.handle_perception_frame(result)

    def on_depth_map(self, frame_id: int, time_s: float, depth_map: np.ndarray) -> None:
        result = self.perception_sync.add_depth_map(frame_id, time_s, depth_map)
        if result is not None:
            self.fusion.handle_perception_frame(result)

    def get_cone_map(self) -> List[Tuple[float, float, float, float, float, float]]:
        return self.fusion.get_cone_map_for_planner()