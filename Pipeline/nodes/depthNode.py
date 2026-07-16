import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelinePaths  # noqa: F401

import cv2
from node import Node
from messages import DepthMessage

# Runs a pluggable depth backend (DepthPro on Mac, DA3 on Jetson). Input is
# converted BGR->RGB before inference, matching main.py. The backend returns a
# metric depth map (H, W) in meters.
class DepthNode(Node):
    def __init__(self, backend, queueSize=8):
        super().__init__('depth', queueSize)
        self.backend = backend

    def process(self, message):
        rgb = cv2.cvtColor(message.image, cv2.COLOR_BGR2RGB)
        depthMap = self.backend.estimateDepth(rgb)
        return DepthMessage(message.frameId, depthMap)
