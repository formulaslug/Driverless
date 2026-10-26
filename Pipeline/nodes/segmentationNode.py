import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelinePaths  # noqa: F401

from node import Node
from messages import SegmentationMessage

# Runs a pluggable segmentation backend (YOLO on Mac, YOLACT-Edge on Jetson).
# The backend returns the shared segmentation dict contract.
class SegmentationNode(Node):
    def __init__(self, backend, queueSize=8):
        super().__init__('segmentation', queueSize)
        self.backend = backend

    def process(self, message):
        result = self.backend.segment(message.image)
        return SegmentationMessage(
            frameId=message.frameId,
            masks=result['masks'],
            boxes=result['boxes'],
            classes=result['classes'],
            confidences=result['confidences'],
            numDetections=result['numDetections'],
        )
