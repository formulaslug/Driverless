from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest

from control_config import M, MAX_STEERING_ANGLE, MAX_TORQUE, R_WHEEL, STEERING_RATIO, T_GEARBOX, WHEELBASE
from controller import Controller
from longitudinal.gain_scheduler import Gains

# Initial PID Gains
K_P = 0.5
K_I = 0.1
K_D = 0.01

# Simulation settings
DT = 0.02
MAX_TIME = 60.0
V_DESIRED = 3.0       # target speed
GOAL_TOLERANCE = 1.0  # stop once within this distance of the final waypoint

# Initial vehicle state
X0, Y0, YAW0, V0 = 0.0, -2.0, 0.0, 0.0  # start slightly off-path on purpose,
                                         # so the controller corrects for it


# Paths

def make_straight_curve(length=20.0, spacing=0.5):
    n = int(length / spacing)
    xs = np.linspace(0, length, n)
    ys = np.zeros(n)
    return list(zip(xs, ys))


def make_s_curve(length=60.0, amplitude=5.0, wavelength=30.0, spacing=0.5):
    n = int(length / spacing)
    xs = np.linspace(0, length, n)
    ys = amplitude * np.sin(2 * np.pi * xs / wavelength)
    return list(zip(xs, ys))


def make_tight_curve(length=60.0, amplitude=5.0, wavelength=30.0, spacing=0.5):
    n = int(length / spacing)
    xs = np.linspace(0, length, n)
    ys = amplitude * np.sin(5 * np.pi * xs / wavelength)
    return list(zip(xs, ys))


def throttle_to_accel(throttle):
    return throttle * MAX_TORQUE * T_GEARBOX / (R_WHEEL * M)


def step_bicycle_model(x, y, yaw, v, delta, a, dt, wheelbase):
    """
    Kinematic bicycle model
    https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357
    """
    x_new = x + v * np.cos(yaw) * dt
    y_new = y + v * np.sin(yaw) * dt
    yaw_new = yaw + (v / wheelbase) * np.tan(delta) * dt
    v_new = max(0.0, v + a * dt)  # floor at 0
    return x_new, y_new, yaw_new, v_new


# Main simulation loop

def run_sim(waypoints):
    controller = Controller(gains=Gains(K_P, K_I, K_D))

    x, y, yaw, v = X0, Y0, YAW0, V0
    t = 0.0

    x_hist, y_hist, v_hist, t_hist, delta_hist = [x], [y], [v], [t], [0.0]

    goal_x, goal_y = waypoints[-1]
    reached = False

    while t < MAX_TIME:
        throttle, steering_angle = controller.update(
            x=x, y=y, yaw=yaw, v=v, t=t, v_desired=V_DESIRED, waypoints=waypoints
        )
        a = throttle_to_accel(throttle)
        delta = np.radians(steering_angle / STEERING_RATIO)

        x, y, yaw, v = step_bicycle_model(x, y, yaw, v, delta, a, DT, WHEELBASE)
        t += DT

        x_hist.append(x)
        y_hist.append(y)
        v_hist.append(v)
        t_hist.append(t)
        delta_hist.append(delta)

        if np.hypot(goal_x - x, goal_y - y) < GOAL_TOLERANCE:
            reached = True
            break

    return {
        "x": x_hist, "y": y_hist, "v": v_hist, "t": t_hist, "delta": delta_hist,
        "waypoints": waypoints, "reached": reached,
    }


def plot_results(result):
    import matplotlib.pyplot as plt

    wp_x = [p[0] for p in result["waypoints"]]
    wp_y = [p[1] for p in result["waypoints"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Trajectory plot
    axes[0].plot(wp_x, wp_y, 'g--', label='reference path')
    axes[0].plot(result["x"], result["y"], 'b-', label='actual trajectory')
    axes[0].scatter([result["x"][0]], [result["y"][0]], c='k', marker='o', label='start')
    axes[0].set_title("Trajectory")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].axis('equal')
    axes[0].legend()
    axes[0].grid(True)

    # Speed tracking plot
    axes[1].plot(result["t"], result["v"], 'b-', label='actual v')
    axes[1].axhline(V_DESIRED, color='g', linestyle='--', label='v_desired')
    axes[1].set_title("Speed Tracking")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("speed (m/s)")
    axes[1].legend()
    axes[1].grid(True)

    # Steering angle over time
    axes[2].plot(result["t"], np.degrees(result["delta"]), 'r-')
    axes[2].set_title("Steering Angle (delta)")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("delta (deg)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# Metrics

def cross_track_errors(result):
    """Distance from each car position to the nearest point on the (densified) path."""
    wp = np.asarray(result["waypoints"], dtype=float)
    dense = np.vstack(
        [np.linspace(a, b, 20, endpoint=False) for a, b in zip(wp[:-1], wp[1:])] + [wp[-1:]]
    )
    car = np.column_stack([result["x"], result["y"]])
    return np.min(np.linalg.norm(car[:, None, :] - dense[None, :, :], axis=2), axis=1)


def settled_cross_track_error(result):
    """Worst cross-track error over the second half of the run (after the initial
    2 m offset has been corrected)."""
    cte = cross_track_errors(result)
    return cte[len(cte) // 2:].max()


# Tests

FEASIBLE_PATHS = {
    "straight": make_straight_curve,
    "s_curve": make_s_curve,
}


@pytest.mark.parametrize("name", FEASIBLE_PATHS)
def test_converges_onto_path_and_reaches_goal(name):
    result = run_sim(FEASIBLE_PATHS[name]())

    assert result["reached"], f"{name}: never reached the final waypoint"
    assert settled_cross_track_error(result) < 0.25, f"{name}: never converged back onto the path"
    assert np.abs(result["delta"]).max() <= MAX_STEERING_ANGLE + 1e-9


def test_speed_settles_at_target_on_s_curve():
    result = run_sim(make_s_curve())
    v = np.asarray(result["v"])

    assert abs(v[-1] - V_DESIRED) < 0.1, f"final speed {v[-1]:.2f} m/s"
    assert v.max() < 1.25 * V_DESIRED, f"overshoot to {v.max():.2f} m/s"


def test_infeasible_tight_curve_stays_stable():
    # The tight curve's radius (~0.7 m) is smaller than the car's minimum turning
    # radius (~3.4 m), so it can't be tracked. Steering should saturate at the
    # limit and the sim should stay finite instead of blowing up.
    result = run_sim(make_tight_curve())

    for key in ("x", "y", "v", "delta"):
        assert np.all(np.isfinite(result[key])), f"{key} went non-finite"
    assert np.abs(result["delta"]).max() <= MAX_STEERING_ANGLE + 1e-9
    assert max(result["v"]) < 1.25 * V_DESIRED


if __name__ == "__main__":
    # waypoints = make_tight_curve()
    # waypoints = make_straight_curve()
    waypoints = make_s_curve()

    result = run_sim(waypoints)
    print("Reached goal" if result["reached"] else f"Hit MAX_TIME ({MAX_TIME}s) without reaching goal")
    plot_results(result)