
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lateral import Lateral
from longitudinal import Longitudinal

from config import K_D_LONGITUDINAL as K_D, K_I_LONGITUDINAL as K_I, K_P_LONGITUDINAL as K_P, K_DD, WHEELBASE, MIN_LOOKAHEAD

# Simple actuator model: throttle/brake (0-1) = signed acceleration (BRAKE_MAX - MAX)
#converts break + throttle into one accel
A_MAX = 3.0    # max acceleration at throttle_output = 1  
A_BRAKE_MAX = 5.0  # max deceleration at brake_output = 1 

# Simulation settings
DT = 0.02          
MAX_TIME = 60.0    
V_DESIRED = 8.0    # target speed
GOAL_TOLERANCE = 1.0  # stop once within this distance of the final waypoint

# Initial vehicle state
X0, Y0, YAW0, V0 = 0.0, -2.0, 0.0, 0.0   # start slightly off-path on purpose,
                                          # so the controller correct for it


#paths


#straight path for PID and accel 
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


#tight curve for autocross
def make_tight_curve(length=60.0, amplitude=5.0, wavelength=30.0, spacing=0.5):
    n = int(length / spacing)
    xs = np.linspace(0, length, n)
    ys = amplitude * np.sin(5 * np.pi * xs / wavelength)
    return list(zip(xs, ys))


# Actuator conversions

def accel_to_throttle_brake(a_des):
    if a_des >= 0:
        throttle = np.clip(a_des, 0, 1)
        brake = 0.0
    else:
        throttle = 0.0
        brake = np.clip(-a_des, 0, 1)
    return throttle, brake


def throttle_brake_to_accel(throttle, brake):
    return throttle * A_MAX - brake * A_BRAKE_MAX



def step_bicycle_model(x, y, yaw, v, delta, a, dt, wheelbase):
    """
    Kinematic bicycle model
    https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357
    """
    x_new = x + v * np.cos(yaw) * dt
    y_new = y + v * np.sin(yaw) * dt
    yaw_new = yaw + (v / wheelbase) * np.tan(delta) * dt
    v_new = v + a * dt
    return x_new, y_new, yaw_new, v_new


# Main simulation loop

def run_sim(waypoints):
    lateral = Lateral(K_dd=K_DD, min_lookahead=MIN_LOOKAHEAD, wheelbase=WHEELBASE)
    longitudinal = Longitudinal(K_p=K_P, K_i=K_I, K_d=K_D)

    x, y, yaw, v = X0, Y0, YAW0, V0
    t = 0.0

    x_hist, y_hist, v_hist, t_hist, delta_hist = [x], [y], [v], [t], [0.0]

    goal_x, goal_y = waypoints[-1]

    while t < MAX_TIME:
        delta = lateral.update(waypoints, x, y, yaw, v)
        a_des = longitudinal.update(v, V_DESIRED, t)

        throttle, brake = accel_to_throttle_brake(a_des)
        a = throttle_brake_to_accel(throttle, brake)

        x, y, yaw, v = step_bicycle_model(x, y, yaw, v, delta, a, DT, WHEELBASE)
        t += DT

        x_hist.append(x)
        y_hist.append(y)
        v_hist.append(v)
        t_hist.append(t)
        delta_hist.append(delta)

        if np.hypot(goal_x - x, goal_y - y) < GOAL_TOLERANCE:
            print(f"Reached goal at t={t:.2f}s")
            break
    else:
        print(f"Hit MAX_TIME ({MAX_TIME}s) without reaching goal")

    return {
        "x": x_hist, "y": y_hist, "v": v_hist, "t": t_hist, "delta": delta_hist,
        "waypoints": waypoints,
    }


def plot_results(result):
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


if __name__ == "__main__":
    waypoints = make_tight_curve()
    #waypoints = make_straight_curve()
    #waypoints = make_s_curve()

    result = run_sim(waypoints)
    plot_results(result)