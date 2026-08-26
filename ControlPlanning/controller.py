from longitudinal import Longitudinal
from lateral import Lateral
from vehichle_dynamics import ThrottleControl
from vehichle_dynamics import SteeringControl
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PathPlanning.path_planner import plan_path

"""
INPUTS:
- Waypoints in smooth_path
- current x, y, yaw, v  
"""

class Controller:
    def __init__(self, K_p, K_i, K_d, K_dd, min_lookahead, wheelbase): #add vd params
        self.longitudinal = Longitudinal(K_p, K_i, K_d)
        self.lateral = Lateral(K_dd, min_lookahead, wheelbase)
        self.throttle_control = ThrottleControl()
        self.steering_control = SteeringControl()

    def update(self, x, y, yaw, v, t, v_desired, waypoints):
        a_des = self.longitudinal.update(v, v_desired, t)
        throttle_output, brake_output = self.throttle_control.accel_to_throttle_brake(a_des, v)
        steer_output = self.steering_control(self.lateral.update(waypoints, x, y, yaw, v))
        return throttle_output, brake_output, steer_output



