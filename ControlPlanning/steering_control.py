from control_config import STEERING_RATIO
import numpy as np

class SteeringControl():
    def __init__(self):
        self.steering_ratio = STEERING_RATIO

    def delta_to_actuator(self, delta):
        steering_wheel_angle = np.degrees(delta) * self.steering_ratio
        return steering_wheel_angle