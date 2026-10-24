import os
import cv2

# Streams frames from a video file with optional target-fps decimation.
# Yields (frameId, frameName, bgrImage) tuples. frameId is the emitted index
# (post-decimation); frameName is frame_%04d of the index in the source video,
# matching SampleData/extractFrames.py naming.
class VideoSource:
    def __init__(self, videoPath, targetFps=None, maxFrames=None, startFrame=0):
        if not os.path.exists(videoPath):
            raise FileNotFoundError(f"Video not found: {videoPath}")

        self.videoPath = videoPath
        self.capture = cv2.VideoCapture(videoPath)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video: {videoPath}")

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.sourceFps = self.capture.get(cv2.CAP_PROP_FPS)
        self.targetFps = targetFps
        self.maxFrames = maxFrames
        self.startFrame = startFrame
        if startFrame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

        if targetFps and self.sourceFps > 0:
            self.frameInterval = max(1, round(self.sourceFps / targetFps))
        else:
            self.frameInterval = 1

        self.outputFps = self.sourceFps / self.frameInterval if self.sourceFps > 0 else targetFps

    def frames(self):
        sourceIndex = 0
        emittedCount = 0

        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break

                if sourceIndex % self.frameInterval == 0:
                    frameName = f'frame_{self.startFrame + sourceIndex:04d}'
                    yield emittedCount, frameName, frame
                    emittedCount += 1

                    if self.maxFrames is not None and emittedCount >= self.maxFrames:
                        break

                sourceIndex += 1
        finally:
            self.release()

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


# Streams frames from a live camera. cameraId is a device index ("0") or any
# string cv2.VideoCapture accepts, such as a GStreamer pipeline on the Jetson.
# Every captured frame is emitted; the stream ends on read failure or maxFrames.
class CameraSource:
    def __init__(self, cameraId, targetFps=None, maxFrames=None):
        source = int(cameraId) if str(cameraId).isdigit() else cameraId
        backend = cv2.CAP_GSTREAMER if isinstance(source, str) and 'appsink' in source else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(source, backend)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open camera: {cameraId}")

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cameraFps = self.capture.get(cv2.CAP_PROP_FPS)
        self.outputFps = cameraFps if cameraFps and cameraFps > 0 else (targetFps or 30.0)
        self.maxFrames = maxFrames

    def frames(self):
        emittedCount = 0
        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break
                yield emittedCount, f'frame_{emittedCount:04d}', frame
                emittedCount += 1
                if self.maxFrames is not None and emittedCount >= self.maxFrames:
                    break
        finally:
            self.release()

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
