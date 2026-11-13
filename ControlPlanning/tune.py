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

'''
varTime = 0
varThrottle = 1
varBrakePressureFront = 2
varBrakePressureRear = 3
varBrakePedalTravel = 4
varSteerAngle = 5
varPosX = 6
varPosY = 7
varPosZ = 8
varVelX = 9
varVelY = 10
varVelZ = 11
varSpeed = 12
varHeadingX = 13
varHeadingY = 14
varHeadingZ = 15
varYawRate = 16
'''

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


state = np.zeros((1000, state_arr.shape[0]))
state[0, :] = state_arr
t = 0
dt = 1 / 100

steps = 1000
gains = Gains(.8, 0.2, 0.05, 0.0)

longitudinal = Longitudinal(K_p=gains.KP, K_i=gains.KI, K_d=gains.KD, Gains=gains)
throttle_map = ThrottleControl().calc_output
v_des = 10.0  # Desired velocity in m/s


for step in range(1, state.shape[0]):
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
plt.show()
