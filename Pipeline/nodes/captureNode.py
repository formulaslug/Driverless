import Pipeline.pipelinePaths  # noqa: F401  adds sibling perception dirs to sys.path

from Pipeline.node import SourceNode
from Pipeline.messages import FrameMessage

# Streams frames from a VideoSource or CameraSource and emits FrameMessage
# carrying the shared CameraModel so every downstream stage uses one camera.
class CaptureNode(SourceNode):
    def __init__(self, source, camera, queueSize=8):
        super().__init__('capture', queueSize)
        self.source = source
        self.camera = camera

    def produce(self):
        for frameId, frameName, image in self.source.frames():
            yield FrameMessage(frameId, frameName, image, self.camera)
