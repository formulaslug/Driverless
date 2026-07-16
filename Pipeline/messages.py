from dataclasses import dataclass
from typing import Optional, List
import numpy as np

@dataclass
class FrameMessage:
    frameId: int
    frameName: str
    image: np.ndarray
    cameraIntrinsics: dict

@dataclass
class DepthMessage:
    frameId: int
    depthMap: np.ndarray

@dataclass
class SegmentationMessage:
    frameId: int
    masks: Optional[np.ndarray]
    boxes: Optional[np.ndarray]
    classes: Optional[list]
    confidences: Optional[np.ndarray]
    numDetections: int

@dataclass
class CalibrationMessage:
    frameId: int
    calibratedScale: float
    planeParams: Optional[np.ndarray]
    inlierRatio: float
    inlierMask: Optional[np.ndarray]

@dataclass
class ConeDistancesMessage:
    frameId: int
    coneDistances: list

@dataclass
class ConeMapMessage:
    frameId: int
    cones: list
    coneMap: list
    filterRef: object

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

# Future IMU input contract. The motion interface accepts these once real
# sensor data is available; unused by the visual estimator today.
@dataclass
class ImuSample:
    timestamp: float
    accel: np.ndarray
    gyro: np.ndarray
    mag: Optional[np.ndarray] = None

@dataclass
class AnnotatedFrameMessage:
    frameId: int
    image: np.ndarray
