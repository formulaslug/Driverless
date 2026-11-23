import numpy as np
import pytest

from tracks import straight_path, arc
from control_config import K_DD, MIN_LOOKAHEAD, WHEELBASE, MAX_STEERING_ANGLE
from lateral.pure_pursuit import PurePursuit


@pytest.fixture
def pp():
    return PurePursuit(K_dd=K_DD, min_lookahead=MIN_LOOKAHEAD, wheelbase=WHEELBASE,
                       max_steer=MAX_STEERING_ANGLE)


def rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(points) @ np.array([[c, -s], [s, c]]).T


def test_straight_ahead_no_steering(pp):
    assert pp.update_delta(straight_path(), 0, 0, 0, 5.0) == pytest.approx(0.0, abs=1e-9)


def test_left_turn_positive_right_turn_negative(pp):
    assert pp.update_delta(arc(10, np.pi / 2, left=True), 0, 0, 0, 5.0) > 0
    assert pp.update_delta(arc(10, np.pi / 2, left=False), 0, 0, 0, 5.0) < 0


def test_known_value_at_min_lookahead(pp):
    # Line at y=0.3, v=0 -> lookahead L=MIN_LOOKAHEAD=2. First point >=2m away is (2, 0.3).
    path = straight_path(y=0.3)
    l_d = np.hypot(2.0, 0.3)
    alpha = np.arctan2(0.3, 2.0)
    expected = np.arctan(2 * np.sin(alpha) / l_d * WHEELBASE)
    assert pp.update_delta(path, 0, 0, 0, 0.0) == pytest.approx(expected)


def test_faster_means_gentler_correction(pp):
    # Same 1m offset: at speed the lookahead is longer so steering is softer
    path = straight_path(length=40, y=1.0)
    slow = pp.update_delta(path, 0, 0, 0, 1.0)
    fast = pp.update_delta(path, 0, 0, 0, 20.0)
    assert 0 < fast < slow


@pytest.mark.parametrize("heading", [np.pi / 2, np.pi, -np.pi / 2, 3.0])
def test_rotating_whole_scene_gives_same_steering(pp, heading):
    # Car + path rotated together should produce identical steering (checks yaw handling)
    path = arc(10, np.pi / 2)
    base = pp.update_delta(path, 0, 0, 0, 5.0)
    rotated = pp.update_delta(rotate(path, heading), 0, 0, heading, 5.0)
    assert rotated == pytest.approx(base, abs=1e-9)


def test_car_offset_in_world_frame(pp):
    # Translating car + path together should not change the answer
    path = arc(10, np.pi / 2)
    base = pp.update_delta(path, 0, 0, 0, 5.0)
    shifted = pp.update_delta(path + [100.0, -50.0], 100.0, -50.0, 0, 5.0)
    assert shifted == pytest.approx(base, abs=1e-9)


def test_steering_clipped_to_limit(pp):
    hairpin = np.array([[1.0, 0.0], [1.5, 2.0], [1.0, 4.0]])
    assert pp.update_delta(hairpin, 0, 0, 0, 5.0) == pytest.approx(MAX_STEERING_ANGLE)


def test_path_shorter_than_lookahead_targets_last_point(pp):
    # Every point is closer than L, so the last point is the target
    short = np.array([[0.5, 0.0], [1.5, 0.2]])
    alpha = np.arctan2(0.2, 1.5)
    expected = np.arctan(2 * np.sin(alpha) / np.hypot(1.5, 0.2) * WHEELBASE)
    assert pp.update_delta(short, 0, 0, 0, 5.0) == pytest.approx(expected)


def test_single_waypoint_and_list_input(pp):
    assert np.isfinite(pp.update_delta(np.array([[5.0, 1.0]]), 0, 0, 0, 5.0))
    assert np.isfinite(pp.update_delta([(x, 0.0) for x in range(10)], 0, 0, 0, 5.0))


def test_car_sitting_on_last_waypoint_does_not_crash(pp):
    path = straight_path()
    assert np.isfinite(pp.update_delta(path, *path[-1], 0, 5.0))
