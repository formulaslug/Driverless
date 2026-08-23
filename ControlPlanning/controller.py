from longitudinal import Longitudinal
from lateral import Lateral
from vehichle_dynamics import VehichleDynamics

class Controller:
    def __init__(self, K_p, K_i, K_d, K_dd, min_lookahead, wheelbase): #add vd params
        self.longitudinal = Longitudinal(K_p, K_i, K_d)
        self.lateral = Lateral(K_dd, min_lookahead, wheelbase)
        self.vehichle_dynamics = VehichleDynamics()

    def update(self, x, y, yaw, v, t, v_desired, waypoints):
        a_des = self.longitudinal.update(v, v_desired, t)
        throttle_output, brake_output = self.vehichle_dynamics.acceleration_to_actuator(a_des, v)
        steer_output = self.lateral.update(waypoints, x, y, yaw, v)
        return throttle_output, brake_output, steer_output