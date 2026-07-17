import os
import cv2

# Streams frames from a video file with optional target-fps decimation.
# Yields (frameId, frameName, bgrImage) tuples. frameId is the emitted index
# (post-decimation), frameName mirrors the legacy frame_%04d.jpg naming so
# downstream outputs stay comparable to the offline frame-glob pipeline.
class VideoSource:
    def __init__(self, videoPath, targetFps=None, maxFrames=None, startFrame=0):
        if not os.path.exists(videoPath):
            raise FileNotFoundError(f"Video not found: {videoPath}")

        self.videoPath = videoPath
        self.capture = cv2.VideoCapture(videoPath)
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video: {videoPath}")

        self.sourceFps = self.capture.get(cv2.CAP_PROP_FPS)
        self.targetFps = targetFps
        self.maxFrames = maxFrames
        self.startFrame = startFrame
        if startFrame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

        if targetFps is not None and self.sourceFps > 0:
            self.frameInterval = max(1, round(self.sourceFps / targetFps))
        else:
            self.frameInterval = 1

        self.outputFps = self.sourceFps / self.frameInterval if self.sourceFps > 0 else targetFps

    def frames(self):
        sourceIndex = 0
        emittedCount = 0

        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            if sourceIndex % self.frameInterval == 0:
                frameName = f'frame_{emittedCount:04d}'
                yield emittedCount, frameName, frame
                emittedCount += 1

                if self.maxFrames is not None and emittedCount >= self.maxFrames:
                    break

            sourceIndex += 1

        self.release()

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
