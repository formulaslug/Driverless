import pytest

from control_config import MAX_ACCEL, MIN_ACCEL
from longitudinal.pid import PID

DT = 0.02


@pytest.fixture
def pid():
    return PID(2.5, 0.6, 0.05)

def test_output_clipped_to_accel_limits():
    assert PID(2.5, 0.6, 0.05).update_des_accel(0.0, 100.0, 0.0) == MAX_ACCEL
    assert PID(2.5, 0.6, 0.05).update_des_accel(100.0, 0.0, 0.0) == MIN_ACCEL


def test_no_integral_windup_while_saturated(pid):
    # Car stuck at 0 m/s with a huge target: output is maxed so the integral must not grow
    for i in range(500):
        pid.update_des_accel(0.0, 50.0, i * DT)
    assert pid.net_integral == 0.0

def test_repeated_timestamp_returns_previous_output(pid):
    a1 = pid.update_des_accel(3.0, 5.0, 1.0)
    a2 = pid.update_des_accel(0.0, 50.0, 1.0)  # same t -> dt=0, must not divide by zero
    assert a2 == a1
