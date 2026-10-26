import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelinePaths  # noqa: F401

from node import Node
from messages import SegmentationMessage, ConeDistancesMessage
from distanceEstimator import DistanceEstimator

# Fuses the bbox-height and flat-ground pinhole estimates into a vehicle-frame
# position per detected cone.
class DistanceNode(Node):
    def __init__(self, camera, queueSize=8):
        super().__init__('distance', queueSize)
        self.estimator = DistanceEstimator(camera)

    def process(self, message):
        if not isinstance(message, SegmentationMessage):
            return None
        if message.boxes is None or message.numDetections == 0:
            coneDistances = []
        else:
            coneDistances = self.estimator.estimateAllCones(message.boxes, message.classes)
        return ConeDistancesMessage(frameId=message.frameId, coneDistances=coneDistances)
