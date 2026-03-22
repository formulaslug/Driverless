
import numpy as np
import polars as pl
from typing import Optional, Tuple, List

from imu_integration import estimate_pose_from_imu
from synchronize import PerceptionSynchronizer
# Daniel's filter module
from Localization.filter import ConeFilter

class EgoMotionEstimator:

    def __init__(self, imu_df: pl.DataFrame):
       
        odometry_df = estimate_pose_from_imu(imu_df)
        
        self.odometry_df = odometry_df.with_columns(
            pl.col("ds").cum_sum().alias("s_cum")
        )
        
        self.times = self.odometry_df["time_s"].to_numpy()
        self.s_cum = self.odometry_df["s_cum"].to_numpy()
        self.theta = self.odometry_df["theta"].to_numpy()
    
    def get_motion_between(self, t_start: float, t_end: float) -> Tuple[float, float, float]:

        s_start = np.interp(t_start, self.times, self.s_cum)
        s_end = np.interp(t_end, self.times, self.s_cum)
        theta_start = np.interp(t_start, self.times, self.theta)
        theta_end = np.interp(t_end, self.times, self.theta)
        
        ds = s_end - s_start
        
        dyaw = theta_end - theta_start
        
        dx = ds * np.cos(theta_start)
        dy = ds * np.sin(theta_start)
        
        return dx, dy, dyaw


class MappingAndFusion:
    def __init__(self, ego_motion: EgoMotionEstimator, cone_filter: ConeFilter):

        self.ego_motion = ego_motion
        self.cone_filter = cone_filter
        self.last_frame_time: Optional[float] = None
    
    def handle_perception_frame(self, frame: dict) -> None:
    
        time_s = frame['time_s']
        cone_measurements = frame['cone_measurements']
        
        detections = [
            (x, y, blue, yellow, sOrange, bOrange)
            for (x, y), (blue, yellow, sOrange, bOrange) in cone_measurements
        ]
        
        if self.last_frame_time is None:
            dx, dy, dyaw = 0.0, 0.0, 0.0
        else:
            dx, dy, dyaw = self.ego_motion.get_motion_between(
                self.last_frame_time,
                time_s
            )
        
        self.cone_filter.update(detections, dx, dy, dyaw)
        
        self.last_frame_time = time_s
    
    def get_cone_map_for_planner(self) -> List[Tuple[float, float, float, float, float, float]]:
        return self.cone_filter.getConeMap()


class PerceptionFusionPipeline:

    def __init__(self, 
                 perception_sync: PerceptionSynchronizer,
                 fusion: MappingAndFusion):
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