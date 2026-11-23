"""
Mock path planning demo: sine-wave path + sine-wave v_desired, bicycle model.
Run from ControlPlanning/:  python -m tests.closed_loop_demo
"""
import numpy as np
import matplotlib.pyplot as plt

from controller import Controller
from control_config import M, MAX_TORQUE, R_WHEEL, T_GEARBOX, STEERING_RATIO, WHEELBASE

DT = 0.01


class MockPathPlanning:
    """Fake path planner: sine-wave path, hands out the next few waypoints + a desired speed."""
    def __init__(self, length=150.0, amplitude=3.0, wavelength=30.0, window=40):
        xs = np.arange(0.0, length, 0.5)
        self.path = np.column_stack([xs, amplitude * np.sin(2 * np.pi * xs / wavelength)])
        self.window = window

    def plan(self, x, y, t):
        closest = np.argmin(np.linalg.norm(self.path - [x, y], axis=1))
        waypoints = self.path[closest:closest + self.window]
        v_desired = 5.0 + 2.0 * np.sin(2 * np.pi * t / 15.0)

        # Slow down to 0 at the end of the path
        # Use continuous distance left (not waypoint count) so v_desired doesn't staircase
        dist_left = np.linalg.norm(self.path[-1] - [x, y])
        v_desired = min(v_desired, np.sqrt(2 * 1.0 * max(dist_left - 0.5, 0.0)))  # stop 0.5 m short
        return waypoints, v_desired


class BicycleModel:
    """Kinematic bicycle model driven by throttle and steering wheel angle."""
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x, self.y, self.yaw, self.v = x, y, yaw, v

    def step(self, throttle, wheel_deg, dt):
        delta = np.radians(wheel_deg / STEERING_RATIO)
        accel = throttle * MAX_TORQUE * T_GEARBOX / (R_WHEEL * M)
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / WHEELBASE * np.tan(delta) * dt
        self.v = max(0.0, self.v + accel * dt)


def run():
    planner = MockPathPlanning()
    car = BicycleModel(yaw=0)  # start not facing track angle
    controller = Controller()

    t = 0.0
    log = []
    while t < 60.0:  # safety time limit
        waypoints, v_desired = planner.plan(car.x, car.y, t)
        if t > 1.0 and car.v < 0.05 and v_desired < 0.1:
            break  # stopped at end of path

        throttle, wheel_deg = controller.update(car.x, car.y, car.yaw, car.v, t, v_desired, waypoints)
        log.append((t, car.x, car.y, car.v, v_desired, throttle))

        car.step(throttle, wheel_deg, DT)
        t += DT

    return planner.path, np.array(log).T


def plot(path, log):
    t, x, y, v, v_des, throttle = log
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    ax1.plot(path[:, 0], path[:, 1], "k--", label="path")
    ax1.plot(x, y, "b", label="car")
    ax1.set_title("Trajectory"); ax1.legend(); ax1.grid()
    ax2.plot(t, v_des, "g--", label="v_desired")
    ax2.plot(t, v, "b", label="v")
    ax2.set_title("Speed (m/s)"); ax2.legend(); ax2.grid()
    ax3.plot(t, throttle, "r", label="throttle")
    ax3.axhline(0, color="k", lw=0.5)
    ax3.set_title("Throttle"); ax3.set_xlabel("t (s)"); ax3.legend(); ax3.grid()
    ax3.sharex(ax2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    path, log = run()
    plot(path, log)
