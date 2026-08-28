from config import STEERING_RATIO


class SteeringControl():
    def __init__(self):
        self.steering_ratio = STEERING_RATIO

    def delta_to_actuator(self, delta):
        steering_angle = delta * self.steering_ratio
        return steering_angle