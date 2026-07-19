import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'motion'))
import pipelinePaths  # noqa: F401

import cv2
from node import Node
from messages import FrameMessage, DepthMessage, OdometryMessage
from synchronizer import FrameSynchronizer, ReorderBuffer

# Estimates inter-frame vehicle motion. Joins the frame image and its depth map
# per frameId, then feeds them to the motion estimator in strict frame order (a
# reorder buffer absorbs concurrent out-of-order arrival) since ego-motion is
# sequential. Emits OdometryMessage(dx, dy, dyaw).
class MotionNode(Node):
    def __init__(self, estimator, queueSize=8):
        super().__init__('motion', queueSize)
        self.estimator = estimator
        self.sync = FrameSynchronizer(['frame', 'depth'])
        self.reorder = ReorderBuffer()

    def process(self, message):
        if isinstance(message, FrameMessage):
            ready = self.sync.add(message.frameId, 'frame', message)
        elif isinstance(message, DepthMessage):
            ready = self.sync.add(message.frameId, 'depth', message)
        else:
            return None

        if ready is None:
            return None

        outputs = []
        for item in self.reorder.push(message.frameId, ready):
            outputs.append(self._estimate(item))
        return outputs

    def onShutdown(self):
        return [self._estimate(item) for item in self.reorder.flush()]

    def _estimate(self, item):
        frame = item['frame']
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        dx, dy, dyaw = self.estimator.estimate(
            frame.frameId, gray, item['depth'].depthMap, frame.cameraIntrinsics
        )
        return OdometryMessage(frame.frameId, dx, dy, dyaw)
