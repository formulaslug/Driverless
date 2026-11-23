import pytest

from control_config import M, R_WHEEL, T_GEARBOX, MAX_TORQUE, MAX_ACCEL, MIN_ACCEL
from throttle_control import ThrottleControl

THROTTLE_MIN, THROTTLE_MAX = -0.2, 1.0


@pytest.fixture
def tc():
    return ThrottleControl()


def test_zero_accel_is_zero_throttle(tc):
    assert tc.calc_output(0.0) == 0.0


def test_sign_follows_accel(tc):
    assert tc.calc_output(1.0) > 0
    assert tc.calc_output(-1.0) < 0


def test_known_value(tc):
    # throttle = (m * a * r_wheel / gear_ratio) / max_torque
    expected = M * 1.0 * R_WHEEL / T_GEARBOX / MAX_TORQUE
    assert tc.calc_output(1.0) == pytest.approx(expected)


@pytest.mark.parametrize("a_des", [-1000.0, -5.0, 0.0, 5.0, 1000.0])
def test_output_always_in_range(tc, a_des):
    assert THROTTLE_MIN <= tc.calc_output(a_des) <= THROTTLE_MAX


def test_extremes_saturate(tc):
    assert tc.calc_output(1000.0) == THROTTLE_MAX
    assert tc.calc_output(-1000.0) == THROTTLE_MIN


def test_pid_limits_fit_inside_throttle_range(tc):
    # If these fail, the PID's accel limits are being silently cut off by the throttle clip
    assert tc.calc_output(MAX_ACCEL) <= THROTTLE_MAX
    assert tc.calc_output(MIN_ACCEL) >= THROTTLE_MIN
