from dataclasses import dataclass
from typing import Optional, List
import numpy as np

@dataclass
class FrameMessage:
    frameId: int
    frameName: str
    image: np.ndarray
    camera: object

@dataclass
class SegmentationMessage:
    frameId: int
    masks: Optional[np.ndarray]
    boxes: Optional[np.ndarray]
    classes: Optional[list]
    confidences: Optional[np.ndarray]
    numDetections: int

@dataclass
class ConeDistancesMessage:
    frameId: int
    coneDistances: list

@dataclass
class ConeMapMessage:
    frameId: int
    cones: list
    coneMap: list

@dataclass
class PathMessage:
    frameId: int
    smoothPath: Optional[np.ndarray]
    curvature: Optional[np.ndarray]

@dataclass
class OdometryMessage:
    frameId: int
    dx: float
    dy: float
    dyaw: float

# IMU input contract. ImuMotionEstimator accepts these once real sensor data is available.
@dataclass
class ImuSample:
    timestamp: float
    accel: np.ndarray
    gyro: np.ndarray
    mag: Optional[np.ndarray] = None

# Emitted by a node whose process() raised so join nodes and reorder buffers can
# discard the frame instead of waiting on it forever.
@dataclass
class FailedFrameMessage:
    frameId: int
    source: str
