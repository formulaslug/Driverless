from longitudinal.longitudinal import Longitudinal
from lateral.lateral import Lateral
from steering_control import SteeringControl
"""
INPUTS:
- Waypoints in smooth_path
- current x, y, yaw, v, t 
- desired velocity (m/s)
- config params for longitudinal and lateral controllers
OUTPUTS:
- desired acceleration(m/s^2), curvature (1/m)
"""

class Controller:
    def __init__(self, K_p, K_i, K_d, K_dd, min_lookahead, wheelbase): #add vd params
        self.longitudinal = Longitudinal(K_p, K_i, K_d)
        self.lateral = Lateral(K_dd, min_lookahead, wheelbase)
        self.steering_control = SteeringControl()

    def update(self, x, y, yaw, v, t, v_desired, waypoints):
        a_des = self.longitudinal.update(v, v_desired, t)
        curvature = self.lateral.pure_pursuit.get_curvature(waypoints, x, y, yaw, v)
        return a_des, curvature



