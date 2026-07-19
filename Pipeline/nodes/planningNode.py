import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelinePaths  # noqa: F401

import numpy as np
from node import Node
from messages import ConeMapMessage, PathMessage
from coneLocalizer import coneFilterToPathPlannerInput
from path_planner import plan_path

# Plans a path from the current cone map. The map is already expressed in the
# current vehicle frame (ConeFilter re-frames it each update), so the vehicle is
# at the origin heading +X. Emits PathMessage(smoothPath, curvature); both None
# when there are too few cones for a valid path.
class PlanningNode(Node):
    def __init__(self, queueSize=8):
        super().__init__('planning', queueSize)

    def process(self, message):
        smoothPath, curvature = None, None
        plannerInput = coneFilterToPathPlannerInput(message.cones)
        if plannerInput is not None:
            positions, coordinateConfidence, colors = plannerInput
            smoothPath, curvature = plan_path(
                positions, coordinateConfidence, colors, np.array([0.0, 0.0]), 0.0
            )
        return PathMessage(frameId=message.frameId, smoothPath=smoothPath, curvature=curvature)
