# not currently used in controller


from config import R_WHEEL,GR, MASS
import numpy as np

class ThrottleControl():
    def __init__(self, ):
        self.r_wheel = R_WHEEL
        self.gr = GR
        self.m = MASS

    def calc_output(self, a_des):
        throttle = self.calc_throttle(a_des)
        return throttle

    def calc_brake_pressure(self, a_des):
        pass
    
    def calc_throttle(self, a_des):
        F = self.m * a_des
        torque = F * self.r_wheel
        motor_torque = torque / self.gr
        throttle = motor_torque / 180
        throttle = np.clip(throttle, -0.2, 1)
        return throttle
    