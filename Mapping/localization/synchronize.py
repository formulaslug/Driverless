from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional
from yolact_integration import process_frame, get_cone_measurements_for_kalman


@dataclass
class PerceptionFrame:
    frame_id: int
    time_s: float
    yolact_output: dict
    depth_map: np.ndarray


class PerceptionSynchronizer:

    def __init__(self, min_confidence: float = 0.7, use_masks: bool = True):
        self.min_confidence = min_confidence
        self.use_masks = use_masks
        
        self._yolact_buffer: Dict[int, dict] = {}  # frame_id -> (time_s, yolact_output)
        self._depth_buffer: Dict[int, dict] = {}   # frame_id -> (time_s, depth_map)
    
    def add_yolact_result(self, frame_id: int, time_s: float, yolact_output: dict) -> Optional[dict]:

        if frame_id in self._depth_buffer:
            depth_data = self._depth_buffer.pop(frame_id)
            depth_time_s = depth_data['time_s']
            depth_map = depth_data['depth_map']
            
            return self._process_complete_frame(frame_id, time_s, yolact_output, depth_map)
        
        else:
            self._yolact_buffer[frame_id] = {
                'time_s': time_s,
                'yolact_output': yolact_output
            }
            return None
    
    def add_depth_map(self, frame_id: int, time_s: float, depth_map: np.ndarray) -> Optional[dict]:
       
        if frame_id in self._yolact_buffer:
            yolact_data = self._yolact_buffer.pop(frame_id)
            yolact_time_s = yolact_data['time_s']
            yolact_output = yolact_data['yolact_output']
            
            return self._process_complete_frame(frame_id, yolact_time_s, yolact_output, depth_map)
        
        else:
            self._depth_buffer[frame_id] = {
                'time_s': time_s,
                'depth_map': depth_map
            }
            return None
    
    def _process_complete_frame(self, frame_id: int, time_s: float, 
                               yolact_output: dict, depth_map: np.ndarray) -> dict:
        
        detections_df = process_frame(
            yolact_output=yolact_output,
            depth_map=depth_map,
            min_confidence=self.min_confidence,
            use_masks=self.use_masks
        )
        
        cone_measurements = get_cone_measurements_for_kalman(detections_df)
        
        return {
            'frame_id': frame_id,
            'time_s': time_s,
            'cone_measurements': cone_measurements
        }
    
    def get_buffer_status(self) -> dict:
        return {
            'pending_yolact': len(self._yolact_buffer),
            'pending_depth': len(self._depth_buffer)
        }
    
    def clear_buffers(self):
        self._yolact_buffer.clear()
        self._depth_buffer.clear()
