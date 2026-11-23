import numpy as np
import pytest

from tracks import straight_path, arc
from control_config import K_DD, MIN_LOOKAHEAD, WHEELBASE, MAX_STEERING_ANGLE
from lateral.pure_pursuit import PurePursuit


def rotate(path, angle):
    """Rotate an (N, 2) path about the origin by angle (rad)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(path) @ np.array([[c, s], [-s, c]])

@pytest.fixture
def pp():
    return PurePursuit(K_dd=K_DD, min_lookahead=MIN_LOOKAHEAD, wheelbase=WHEELBASE,
                       max_steer=MAX_STEERING_ANGLE)

def test_left_turn_positive_right_turn_negative(pp):
    assert pp.update_delta(arc(10, np.pi / 2, left=True), 0, 0, 0, 5.0) > 0
    assert pp.update_delta(arc(10, np.pi / 2, left=False), 0, 0, 0, 5.0) < 0

@pytest.mark.parametrize("heading", [np.pi / 2, np.pi, -np.pi / 2, 3.0])
def test_rotating_whole_scene_gives_same_steering(pp, heading):
    # Car + path rotated together should produce identical steering (checks yaw handling)
    path = arc(10, np.pi / 2)
    base = pp.update_delta(path, 0, 0, 0, 5.0)
    rotated = pp.update_delta(rotate(path, heading), 0, 0, heading, 5.0)
    assert rotated == pytest.approx(base, abs=1e-9)

