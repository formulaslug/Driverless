from longitudinal.longitudinal import Longitudinal
from lateral.lateral import Lateral
from steering_control import SteeringControl
from throttle_control import ThrottleControl
from longitudinal.gain_scheduler import Gains
import numpy as np

"""
INPUTS:
- Waypoints in smooth_path
- current x, y, yaw, v, t 
- desired velocity (m/s)
- config params for longitudinal and lateral controllers
OUTPUTS:
- throttle, steering_wheel_deg
"""

class Controller:
    def __init__(self, gains=Gains(2.5, 0.6, 0.05)): 
        self.longitudinal = Longitudinal(gains)
        self.lateral = Lateral()
        self.steering_control = SteeringControl()
        self.throttle_control = ThrottleControl()

    def update(self, x, y, yaw, v, t, v_desired, waypoints):
        if waypoints is None or len(waypoints) == 0:
            a_des = self.longitudinal.update_des_accel(v, 0.0, t)
            throttle = min(self.throttle_control.calc_output(a_des), 0.0) # never accelerate without a path
            return throttle, 0.0

        if v_desired is None or not np.isfinite(v_desired):
            v_desired = v

        a_des = self.longitudinal.update_des_accel(v, v_desired, t)
        delta = self.lateral.update(waypoints, x, y, yaw, v)
        throttle = self.throttle_control.calc_output(a_des)
        steering_angle = self.steering_control.delta_to_actuator(delta)
        return throttle, steering_angle
