import os
import sys
import traceback
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelinePaths  # noqa: F401

from node import Node
from messages import SegmentationMessage, ConeDistancesMessage, OdometryMessage, ConeMapMessage, FailedFrameMessage
from synchronizer import FrameSynchronizer, ReorderBuffer
from coneLocalizer import perceptionToDetections
from filter import ConeFilter


# Lightweight, thread-safe copy of a cone's planner-relevant state. Snapshotting
# avoids the planner thread reading the live ConeFilter while localization mutates
# it on the next frame; exposes exactly what coneFilterToPathPlannerInput reads.
class ConeSnapshot:
    def __init__(self, cone):
        self.x = cone.x
        self.y = cone.y
        self.cov = cone.cov.copy()
        self.colorConfidences = list(cone.colorConfidences)


# Owns the persistent ConeFilter. Joins distances + segmentation per frameId,
# then applies each frame in strict order once its odometry has arrived (the
# filter is sequential). A frame whose perception failed still has its odometry
# applied, so the map stays aligned with the car. Emits a ConeMapMessage carrying
# a cone snapshot + flat cone map for the planner and the visualizer.
class LocalizationNode(Node):
    def __init__(self, queueSize=8):
        super().__init__('localization', queueSize)
        self.filter = ConeFilter()
        self.sync = FrameSynchronizer(['distances', 'seg'])
        self.reorder = ReorderBuffer()
        self.odometry = {}
        self.waiting = deque()

    def process(self, message):
        if isinstance(message, OdometryMessage):
            self.odometry[message.frameId] = message
        elif isinstance(message, (ConeDistancesMessage, SegmentationMessage)):
            key = 'distances' if isinstance(message, ConeDistancesMessage) else 'seg'
            ready = self.sync.add(message.frameId, key, message)
            if ready is not None:
                self.waiting.extend(self.reorder.push(message.frameId, {'frameId': message.frameId, **ready}))
        else:
            return None
        return self._drain()

    def _handleFailedFrame(self, failure):
        if failure.source == 'motion':
            self.odometry.setdefault(failure.frameId, OdometryMessage(failure.frameId, 0.0, 0.0, 0.0))
        super()._handleFailedFrame(failure)

    def skipFrame(self, frameId):
        self.sync.drop(frameId)
        self.waiting.extend(self.reorder.push(frameId, {'frameId': frameId}))
        return self._drain()

    def onShutdown(self):
        self.waiting.extend(self.reorder.flush())
        return self._drain(force=True)

    def _drain(self, force=False):
        outputs = []
        while self.waiting and (force or self.waiting[0]['frameId'] in self.odometry):
            item = self.waiting.popleft()
            frameId = item['frameId']
            odom = self.odometry.pop(frameId, None) or OdometryMessage(frameId, 0.0, 0.0, 0.0)
            try:
                output = self._updateFilter(item, odom)
            except Exception:
                print(f"[{self.name}] error updating filter for frame {frameId}", file=sys.stderr)
                traceback.print_exc()
                output = FailedFrameMessage(frameId, self.name)
            if output is not None:
                outputs.append(output)
        return outputs

    def _updateFilter(self, item, odom):
        if 'seg' not in item:
            self.filter.update([], odom.dx, odom.dy, odom.dyaw)
            return None

        seg = item['seg']
        detections = perceptionToDetections(
            seg.boxes, seg.classes, seg.confidences, item['distances'].coneDistances
        )
        self.filter.update(detections, odom.dx, odom.dy, odom.dyaw)

        cones = [ConeSnapshot(c) for c in self.filter.cones]
        return ConeMapMessage(
            frameId=seg.frameId,
            cones=cones,
            coneMap=self.filter.getConeMap(),
        )
