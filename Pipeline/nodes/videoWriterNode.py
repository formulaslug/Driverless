import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelinePaths  # noqa: F401

import cv2
import numpy as np
from node import Node
from messages import FrameMessage, SegmentationMessage, ConeDistancesMessage, PathMessage, ConeMapMessage
from synchronizer import FrameSynchronizer, ReorderBuffer

# BGR colors per cone class for overlays.
_CLASS_COLORS = {
    'seg_blue_cone': (255, 128, 0),
    'seg_yellow_cone': (0, 220, 220),
    'seg_orange_cone': (0, 140, 255),
    'seg_large_orange_cone': (0, 90, 255),
    'seg_unknown_cone': (160, 160, 160),
}


# Terminal node: joins frame + segmentation + distances + path + cone map per
# frameId, draws detection boxes with ranges and a bird's-eye inset (cone map +
# planned path), and writes frames to an .mp4 in strict frame order.
class VideoWriterNode(Node):
    def __init__(self, outputPath, fps, bevSize=260, bevScale=6.0, queueSize=8):
        super().__init__('videoWriter', queueSize)
        self.outputPath = outputPath
        self.fps = fps if fps and fps > 0 else 10.0
        self.bevSize = bevSize
        self.bevScale = bevScale
        self.sync = FrameSynchronizer(['frame', 'seg', 'distances', 'path', 'conemap'])
        self.reorder = ReorderBuffer()
        self.writer = None

    def process(self, message):
        keyByType = {
            FrameMessage: 'frame', SegmentationMessage: 'seg', ConeDistancesMessage: 'distances',
            PathMessage: 'path', ConeMapMessage: 'conemap',
        }
        key = keyByType.get(type(message))
        if key is None:
            return None

        ready = self.sync.add(message.frameId, key, message)
        if ready is None:
            return None

        for item in self.reorder.push(message.frameId, ready):
            self._writeFrame(item)
        return None

    def skipFrame(self, frameId):
        self.sync.drop(frameId)
        for item in self.reorder.skip(frameId):
            self._writeFrame(item)
        return None

    def onShutdown(self):
        try:
            for item in self.reorder.flush():
                self._writeFrame(item)
        finally:
            if self.writer is not None:
                self.writer.release()
        return None

    def _writeFrame(self, item):
        frame = item['frame'].image.copy()
        self._drawDetections(frame, item['seg'], item['distances'].coneDistances)
        self._drawBev(frame, item['conemap'].coneMap, item['path'].smoothPath)

        if self.writer is None:
            H, W = frame.shape[:2]
            os.makedirs(os.path.dirname(os.path.abspath(self.outputPath)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.outputPath, fourcc, self.fps, (W, H))
        self.writer.write(frame)

    def _drawDetections(self, frame, seg, coneDistances):
        if seg.boxes is None:
            return
        for i, (box, cls) in enumerate(zip(seg.boxes, seg.classes)):
            x1, y1, x2, y2 = [int(v) for v in box]
            name = cls['name'] if isinstance(cls, dict) else cls
            color = _CLASS_COLORS.get(name, (160, 160, 160))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            distance = coneDistances[i]['distance'] if i < len(coneDistances) else None
            if distance is not None:
                cv2.putText(frame, f"{distance:.1f}m", (x1, max(10, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draws a top-down inset: X forward = up, Y left = left, vehicle at bottom-center.
    def _drawBev(self, frame, coneMap, smoothPath):
        H, W = frame.shape[:2]
        size = min(self.bevSize, H, W)
        bev = np.full((size, size, 3), 30, dtype=np.uint8)
        originX, originY = size // 2, size - 15

        def toPixel(x, y):
            return int(originX - y * self.bevScale), int(originY - x * self.bevScale)

        cv2.drawMarker(bev, (originX, originY), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 12, 2)

        classColorByIndex = [(255, 128, 0), (0, 220, 220), (0, 140, 255), (0, 90, 255)]
        for cone in coneMap:
            x, y = cone[0], cone[1]
            colorConfs = cone[2:6]
            color = classColorByIndex[int(np.argmax(colorConfs))]
            px, py = toPixel(x, y)
            if 0 <= px < size and 0 <= py < size:
                cv2.circle(bev, (px, py), 3, color, -1)

        if smoothPath is not None and len(smoothPath) > 1:
            pts = np.array([toPixel(p[0], p[1]) for p in smoothPath], dtype=np.int32)
            cv2.polylines(bev, [pts], False, (0, 255, 0), 2)

        frame[H - size:H, 0:size] = bev
