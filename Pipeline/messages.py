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
