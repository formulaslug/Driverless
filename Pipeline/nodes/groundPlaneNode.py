import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelinePaths  # noqa: F401

from node import Node
from messages import CalibrationMessage
from ground_plane_ransac import estimateGroundPlane

# Fits a ground plane to the metric depth map via RANSAC. Depth is already
# metric (both DepthPro and DA3 backends output meters), so isMetricDepth=True.
class GroundPlaneNode(Node):
    def __init__(self, cameraIntrinsics, queueSize=8):
        super().__init__('groundPlane', queueSize)
        self.cameraIntrinsics = cameraIntrinsics

    def process(self, message):
        planeParams, inlierRatio, inlierMask = estimateGroundPlane(
            message.depthMap,
            self.cameraIntrinsics,
            exclusionAreas=None,
            subsampleStep=5,
            inlierThreshold=0.1,
            maxTrials=100,
            maxTiltAngle=45,
            isMetricDepth=True,
        )
        return CalibrationMessage(
            frameId=message.frameId,
            calibratedScale=1.0,
            planeParams=planeParams,
            inlierRatio=inlierRatio,
            inlierMask=inlierMask,
        )
