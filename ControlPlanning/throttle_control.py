from control_config import R_WHEEL, T_GEARBOX, M, MAX_TORQUE
import numpy as np

class ThrottleControl():
    def __init__(self, ):
        self.r_wheel = R_WHEEL
        self.gr = T_GEARBOX
        self.m = M
        self.max_torque = MAX_TORQUE

    def calc_output(self, a_des):
        throttle = self.calc_throttle(a_des)
        return throttle

    def calc_brake_pressure(self, a_des):
        pass
    
    def calc_throttle(self, a_des):
        F = self.m * a_des
        torque = F * self.r_wheel
        motor_torque = torque / self.gr
        throttle = motor_torque / self.max_torque
        throttle = np.clip(throttle, -0.2, 1)
        return throttle
    