import numpy as np
import pytest

from control_config import STEERING_RATIO, MAX_STEERING_ANGLE
from controller import Controller
from steering_control import SteeringControl

MAX_WHEEL_DEG = np.degrees(MAX_STEERING_ANGLE) * STEERING_RATIO


@pytest.fixture
def sc():
    return SteeringControl()


def test_zero_delta_is_zero_wheel_angle(sc):
    assert sc.delta_to_actuator(0.0) == 0.0


def test_known_value(sc):
    assert sc.delta_to_actuator(0.1) == pytest.approx(np.degrees(0.1) * STEERING_RATIO)


def test_sign_preserved(sc):
    assert sc.delta_to_actuator(0.2) > 0
    assert sc.delta_to_actuator(-0.2) < 0
    assert sc.delta_to_actuator(-0.2) == pytest.approx(-sc.delta_to_actuator(0.2))


def test_controller_never_exceeds_wheel_limit():
    # A path that turns far tighter than the car can: steering must clip at the limit
    hairpin = np.array([[1.0, 0.0], [1.5, 2.0], [1.0, 4.0]])
    for path, sign in [(hairpin, 1), (hairpin * [1, -1], -1)]:
        _, wheel_deg = Controller().update(0, 0, 0, 5.0, 0.0, 5.0, path)
        assert wheel_deg == pytest.approx(sign * MAX_WHEEL_DEG)
