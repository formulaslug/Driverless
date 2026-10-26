import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'motion'))
import pipelinePaths  # noqa: F401

from node import Node
from messages import FrameMessage, OdometryMessage

# Emits one OdometryMessage(dx, dy, dyaw) per frame from the motion estimator.
# Frames arrive in order from the single capture upstream, so the estimator
# sees them sequentially.
class MotionNode(Node):
    def __init__(self, estimator, queueSize=8):
        super().__init__('motion', queueSize)
        self.estimator = estimator

    def process(self, message):
        if not isinstance(message, FrameMessage):
            return None
        dx, dy, dyaw = self.estimator.estimate(message.frameId)
        return OdometryMessage(message.frameId, dx, dy, dyaw)
