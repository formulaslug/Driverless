from longitudinal.longitudinal import Longitudinal
from lateral.lateral import Lateral
from steering_control import SteeringControl
from config import K_DD, MIN_LOOKAHEAD, WHEELBASE
"""
INPUTS:
- Waypoints in smooth_path
- current x, y, yaw, v, t 
- desired velocity (m/s)
- config params for longitudinal and lateral controllers
OUTPUTS:
- 
"""

class Controller:
    def __init__(self, K_p, K_i, K_d, K_dd = K_DD, min_lookahead = MIN_LOOKAHEAD, wheelbase = WHEELBASE): 
        self.longitudinal = Longitudinal(K_p = K_p, K_i = K_i, K_d = K_d)
        self.lateral = Lateral(K_dd = K_dd, min_lookahead = min_lookahead, wheelbase = wheelbase)
        self.steering_control = SteeringControl()

    def update(self, x, y, yaw, v, t, v_desired, waypoints):
        a_des = self.longitudinal.update_des_accel(v, v_desired, t)
        curvature = self.lateral.pure_pursuit.update(waypoints, x, y, yaw, v)
        return a_des, curvature
