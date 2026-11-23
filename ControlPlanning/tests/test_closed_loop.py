"""
Drives the full Controller around simple tracks with a kinematic bicycle model.
This is the "does the whole thing actually work" test: catches sign errors,
degree/radian mixups and an unstable PID.
"""
import numpy as np
import pytest

from tracks import straight_path, s_curve, arc
from control_config import M, MAX_TORQUE, R_WHEEL, T_GEARBOX, STEERING_RATIO, WHEELBASE, MAX_STEERING_ANGLE
from controller import Controller

DT = 0.02
MAX_TIME = 60.0
GOAL_TOLERANCE = 1.0


def throttle_to_accel(throttle):
    return throttle * MAX_TORQUE * T_GEARBOX / (R_WHEEL * M)


def simulate(waypoints, v_desired, x0=0.0, y0=0.0, yaw0=0.0, v0=0.0):
    controller = Controller()
    x, y, yaw, v, t = x0, y0, yaw0, v0, 0.0
    hist = {"x": [x], "y": [y], "v": [v], "delta": [0.0], "reached": False}

    while t < MAX_TIME:
        throttle, wheel_deg = controller.update(x, y, yaw, v, t, v_desired, waypoints)
        delta = np.radians(wheel_deg / STEERING_RATIO)
        a = throttle_to_accel(throttle)

        x += v * np.cos(yaw) * DT
        y += v * np.sin(yaw) * DT
        yaw += v / WHEELBASE * np.tan(delta) * DT
        v = max(0.0, v + a * DT)
        t += DT

        hist["x"].append(x); hist["y"].append(y); hist["v"].append(v); hist["delta"].append(delta)
        if np.hypot(*(np.asarray(waypoints[-1]) - [x, y])) < GOAL_TOLERANCE:
            hist["reached"] = True
            break
    return hist


def cross_track_errors(hist, waypoints):
    """Distance from each car position to the nearest point on the (densified) path."""
    wp = np.asarray(waypoints)
    dense = np.vstack([np.linspace(a, b, 20, endpoint=False) for a, b in zip(wp[:-1], wp[1:])] + [wp[-1:]])
    car = np.column_stack([hist["x"], hist["y"]])
    return np.min(np.linalg.norm(car[:, None, :] - dense[None, :, :], axis=2), axis=1)


TRACKS = {
    "straight": (straight_path(40), 5.0),
    "s_curve": (s_curve(), 4.0),
    "hairpin_left": (arc(5.0, np.pi), 3.0),
    "hairpin_right": (arc(5.0, np.pi, left=False), 3.0),
}


@pytest.mark.parametrize("name", TRACKS)
def test_follows_track(name):
    path, v_des = TRACKS[name]
    hist = simulate(path, v_des)

    assert hist["reached"], f"{name}: never reached the end of the path"
    assert cross_track_errors(hist, path).max() < 0.75, f"{name}: drifted too far off the path"
    assert np.abs(hist["delta"]).max() <= MAX_STEERING_ANGLE + 1e-9


@pytest.mark.parametrize("name", ["straight", "s_curve"])
def test_speed_settles_at_target(name):
    path, v_des = TRACKS[name]
    hist = simulate(path, v_des)
    v = np.array(hist["v"])
    overshoot = v.max() - v_des
    assert overshoot < 0.05 * v_des + 0.1, f"overshoot {overshoot:.2f} m/s"
    assert abs(v[-1] - v_des) < 0.3


def test_recovers_from_starting_off_path():
    path, v_des = straight_path(40), 5.0
    hist = simulate(path, v_des, y0=-2.0)
    cte = cross_track_errors(hist, path)
    assert hist["reached"]
    assert cte[len(cte) // 2:].max() < 0.2, "did not converge back onto the path"


def test_recovers_from_bad_heading():
    path, v_des = straight_path(40), 5.0
    hist = simulate(path, v_des, yaw0=np.radians(30))
    cte = cross_track_errors(hist, path)
    assert hist["reached"]
    assert cte[len(cte) // 2:].max() < 0.2


def test_losing_the_path_brakes_to_a_stop():
    controller = Controller()
    v, t = 5.0, 0.0
    for _ in range(int(10 / DT)):
        throttle, wheel = controller.update(0, 0, 0, v, t, 5.0, None)
        assert throttle <= 0.0 and wheel == 0.0
        v = max(0.0, v + throttle_to_accel(throttle) * DT)
        t += DT
    assert v == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("bad_path", [None, np.empty((0, 2)), []])
def test_no_path_does_not_crash(bad_path):
    throttle, wheel = Controller().update(0, 0, 0, 3.0, 0.0, 5.0, bad_path)
    assert throttle <= 0.0
    assert wheel == 0.0
