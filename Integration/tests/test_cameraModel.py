import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cameraModel import CameraModel


def projectGroundPoint(camera, forward, left):
    p = camera.pitchRad
    relX = forward - camera.offsetX
    relY = left - camera.offsetY
    zc = relX * np.cos(p) + camera.mountHeight * np.sin(p)
    yc = -relX * np.sin(p) + camera.mountHeight * np.cos(p)
    return camera.cx - camera.fx * relY / zc, camera.cy + camera.fy * yc / zc


@pytest.mark.parametrize('pitchDeg', [0.0, 10.0, -5.0])
def testGroundPointRoundTrip(pitchDeg):
    camera = CameraModel(640, 480, 500, 500, 320, 240, 1.0, np.radians(pitchDeg))
    for forward, left in [(5.0, 0.0), (10.0, 1.5), (20.0, -2.0)]:
        u, v = projectGroundPoint(camera, forward, left)
        x, y = camera.groundPoint(u, v)
        assert x == pytest.approx(forward, abs=1e-6)
        assert y == pytest.approx(left, abs=1e-6)


@pytest.mark.parametrize('pitchDeg', [0.0, 10.0, -5.0])
def testPointAtDepthMatchesGroundPoint(pitchDeg):
    camera = CameraModel(640, 480, 500, 500, 320, 240, 1.0, np.radians(pitchDeg))
    u, v = projectGroundPoint(camera, 12.0, 1.0)
    t = camera.mountHeight / -camera.rayDirection(u, v)[2]
    x, y = camera.pointAtDepth(u, v, t)
    assert (x, y) == pytest.approx((12.0, 1.0), abs=1e-6)


def testGroundPointAboveHorizonIsNone():
    camera = CameraModel(640, 480, 500, 500, 320, 240, 1.0, 0.0)
    assert camera.groundPoint(320, 100) is None
