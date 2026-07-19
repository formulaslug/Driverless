import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelinePaths  # noqa: F401

from node import Node
from messages import SegmentationMessage, DepthMessage, CalibrationMessage, ConeDistancesMessage
from synchronizer import FrameSynchronizer
from distanceEstimator import DistanceEstimator

# Joins segmentation, depth, and ground-plane by frameId, then fuses the three
# per-cone distance estimates. Constructed with the shared cameraIntrinsics so
# all distance methods use one intrinsics source (fixes the camera.json mismatch).
class DistanceNode(Node):
    def __init__(self, cameraIntrinsics, queueSize=8):
        super().__init__('distance', queueSize)
        self.estimator = DistanceEstimator(cameraIntrinsics=cameraIntrinsics)
        self.sync = FrameSynchronizer(['seg', 'depth', 'calib'])

    def process(self, message):
        if isinstance(message, SegmentationMessage):
            key, value = 'seg', message
        elif isinstance(message, DepthMessage):
            key, value = 'depth', message
        elif isinstance(message, CalibrationMessage):
            key, value = 'calib', message
        else:
            return None

        ready = self.sync.add(message.frameId, key, value)
        if ready is None:
            return None

        seg = ready['seg']
        if seg.boxes is None or seg.numDetections == 0:
            coneDistances = []
        else:
            coneDistances = self.estimator.estimateAllCones(
                seg.boxes, seg.classes, ready['depth'].depthMap, ready['calib'].planeParams
            )
        return ConeDistancesMessage(frameId=message.frameId, coneDistances=coneDistances)
