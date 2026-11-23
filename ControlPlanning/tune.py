from throttle_control import ThrottleControl
from longitudinal.gain_scheduler import Gains
from longitudinal.longitudinal import Longitudinal
import numpy as np
import matplotlib.pyplot as plt


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Sims-Data")))
from FullVehicleSim.paramLoader import varSpeed, varThrottle, varTime
from FullVehicleSim.engine import dynamicStepState

gains = Gains(2.5, 0.6, 0.05)
v_des = 12
steps = 2000
dt = 1 / 100

state_arr = np.array(
        [0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 1, 0, 
        0, 0, 25, 25, 1, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 80000, 0, 120, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 180, 0, 0]
    )

# Tuned for 10m/s
# Rise time target: 10% - 90% (.9 - .1) / MAX ACCEL(3.0)
# Overshoot: 5% cap

# Tune from start
def run_tune(gains = gains, v_des = v_des, steps = steps):
    state = np.zeros((steps, state_arr.shape[0]))
    state[0, :] = state_arr
    t = 0

    longitudinal = Longitudinal(gains=gains)
    throttle_map = ThrottleControl().calc_output


    for step in range(1, steps):
        prev = state[step - 1]
        a_des = longitudinal.update_des_accel(prev[varSpeed], v_des, t)
        throttle = throttle_map(a_des)

        prev[varTime] = t
        prev[varThrottle] = throttle
        state[step] = dynamicStepState(prev)
        t += dt

    pid = longitudinal.pid
    rise_str = f"{pid.rise_time:.3f} s" if pid.rise_time_defined else "Not reached"
    settle_str = f"{pid.settle_time:.3f} s" if pid.settle_time is not None else "Did not settle"

    print(f"Gains | KP: {gains.KP}, KI: {gains.KI}, KD: {gains.KD}")
    print(f"Rise Time:   {rise_str}")
    print(f"Settle Time: {settle_str}")
    print(f"Overshoot:   {pid.overshoot:.2f}%")
        
    times = np.arange(steps) * dt
    plt.plot(times, state[:, varSpeed])
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Time')
    plt.grid()

# Use to calc rise_time
def run_open_loop(throttle_fn, n_steps=1500):
    """
    Builds state, gets velocities, builds acceleration. 
    """
    s = np.zeros((n_steps, state_arr.shape[0]))
    s[0, :] = state_arr
    t = 0.0
    for step in range(1, n_steps):
        prev = s[step - 1]
        prev[varTime] = t
        prev[varThrottle] = throttle_fn(t, prev[varSpeed])
        s[step] = dynamicStepState(prev)
        t += dt
    v = s[:, varSpeed]
    a = np.gradient(v, dt)
    return v, a

def plot_open():
    v, a = run_open_loop(lambda t, v: 1.0) # Use max throttle at all velocities and times, to find peak acceleration
    print(f"Peak accel: {a[10:-10].max():.2f} m/s^2")
    plt.figure(); plt.plot(v, a)
    plt.xlabel("Speed (m/s)"); plt.ylabel("Accel (m/s^2)"); plt.grid()
if __name__ == "__main__":
    run_tune() 
    plt.show()
