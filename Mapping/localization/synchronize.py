from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional, Tuple
from yolact_integration import process_frame, get_cone_measurements_for_kalman

@dataclass
class PerceptionFrame:
    frame_id: int
    time_s: float
    yolact_output: dict
    depth_map: np.ndarray

# avoids the extra wrapping dict
_YolactEntry = Tuple[float, dict] # (time_s, yolact_output)
_DepthEntry  = Tuple[float, np.ndarray]  #(time_s, depth_map)

class PerceptionSynchronizer:

    def __init__(self, min_confidence: float = 0.7, use_masks: bool = True):
        if not (0.0 <= min_confidence <=  1.0):
            raise ValueError(f"min_confidence must be in [0, 1] , get {min_confidence}")

        self.min_confidence = min_confidence
        self.use_masks = use_masks

        self._yolact_buffer: Dict[int, _YolactEntry] = {}
        self._depth_buffer:  Dict[int, _DepthEntry]  = {}

    def add_yolact_result(self, frame_id: int, time_s: float, yolact_output: dict) -> Optional[dict]:
        if not isinstance(frame_id, int):
            raise TypeError(f"frame_id must be an int, got {type(frame_id)}")
        if not isinstance(yolact_output, dict):
            raise TypeError("yolact_output must be a dict.")

        depth_entry = self._depth_buffer.pop(frame_id, None)
        if depth_entry is not None:
            _,  depth_map = depth_entry
            return self._process_complete_frame(frame_id, time_s, yolact_output, depth_map)

        self._yolact_buffer[frame_id] = (time_s, yolact_output)
        return None

    def add_depth_map(self, frame_id: int, time_s: float, depth_map: np.ndarray) -> Optional[dict]:
        if not isinstance(frame_id, int):
            raise TypeError(f"frame_id must be an int, got {type(frame_id)}")
        if not isinstance(depth_map, np.ndarray) or depth_map.ndim !=  2:
            raise ValueError("depth_map must be a 2-D numpy array")

        yolact_entry = self._yolact_buffer.pop(frame_id, None)
        if yolact_entry is not None:
            yolact_time_s, yolact_output = yolact_entry
            return self._process_complete_frame(frame_id, yolact_time_s, yolact_output, depth_map)

        self._depth_buffer[frame_id] = (time_s, depth_map)
        return None

    def get_buffer_status(self) -> dict:
        return {
            "pending_yolact": len(self._yolact_buffer),
            "pending_depth":  len(self._depth_buffer),
        }

    def clear_buffers(self):
        self._yolact_buffer.clear()
        self._depth_buffer.clear()

    def _process_complete_frame(self, frame_id: int, time_s: float, yolact_output: dict, depth_map: np.ndarray) -> dict:
        detections_df = process_frame(
            yolact_output=yolact_output,
            depth_map=depth_map,
            min_confidence=self.min_confidence,
            use_masks=self.use_masks,
        )

        cone_measurements = get_cone_measurements_for_kalman(detections_df)

        if not isinstance(cone_measurements, list):
            raise TypeError("get_cone_measurements_for_kalman must return a list")

        return {
            "frame_id":         frame_id,
            "time_s":           time_s,
            "cone_measurements": cone_measurements,
        }