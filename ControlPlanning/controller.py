from longitudinal import Longitudinal
from lateral import Lateral
from throttle_control import ThrottleControl
from steering_control import SteeringControl
"""
INPUTS:
- Waypoints in smooth_path
- current x, y, yaw, v 
- config params for longitudinal and lateral controllers
OUTPUTS:
- desired acceleration, steering wheel output 
"""

class Controller:
    def __init__(self, K_p, K_i, K_d, K_dd, min_lookahead, wheelbase): #add vd params
        self.longitudinal = Longitudinal(K_p, K_i, K_d)
        self.lateral = Lateral(K_dd, min_lookahead, wheelbase)
        self.throttle_control = ThrottleControl()
        self.steering_control = SteeringControl()

    def update(self, x, y, yaw, v, t, v_desired, waypoints):
        a_des = self.longitudinal.update(v, v_desired, t)
        steer_output = self.steering_control.delta_to_actuator(self.lateral.update(waypoints, x, y, yaw, v), max_steering_angle=30)
        return a_des, steer_output



