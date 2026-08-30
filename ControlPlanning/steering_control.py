from config import STEERING_RATIO


class SteeringControl():
    def __init__(self):
        self.steering_ratio = STEERING_RATIO

    def delta_to_actuator(self, delta, max_steering_angle):
        steering_angle = delta * self.steering_ratio
        return np.clip(steering_angle, -max_steering_angle, max_steering_angle)