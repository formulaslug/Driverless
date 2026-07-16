import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelinePaths  # noqa: F401  adds sibling perception dirs to sys.path

from node import SourceNode
from messages import FrameMessage
from videoSource import VideoSource
from ground_plane_ransac import getCameraIntrinsics

# Streams frames from a video file and emits FrameMessage. Camera intrinsics are
# derived from the actual frame size (not camera.json) so every downstream stage
# shares one intrinsics source, matching how main.py builds them.
class CaptureNode(SourceNode):
    def __init__(self, videoPath, fov=90, targetFps=None, maxFrames=None, queueSize=8):
        super().__init__('capture', queueSize)
        self.source = VideoSource(videoPath, targetFps=targetFps, maxFrames=maxFrames)
        self.fov = fov
        self.cameraIntrinsics = None

    def produce(self):
        for frameId, frameName, image in self.source.frames():
            if self.cameraIntrinsics is None:
                H, W = image.shape[:2]
                self.cameraIntrinsics = getCameraIntrinsics(W, H, fov=self.fov)
            yield FrameMessage(frameId, frameName, image, self.cameraIntrinsics)
