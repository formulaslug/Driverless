from config import STEERING_RATIO
import numpy as np


# clipping as first iteration
class ThrottleControl():
    def __init__(self):
        pass
    def accel_to_throttle_brake(self, a_des, v = 0): # velocity unused for now
        if a_des >= 0:
            throttle_output = np.clip(a_des, 0, 1)
            brake_output = 0
        else:
            throttle_output = 0
            brake_output = np.clip(-a_des, 0, 1)
        return throttle_output, brake_output



class SteeringControl():
    def __init__(self):
        self.steering_ratio = STEERING_RATIO

    def delta_to_actuator(self, delta):
        steering_angle = delta * self.steering_ratio
        return steering_angle