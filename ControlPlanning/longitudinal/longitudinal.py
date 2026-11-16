from longitudinal.pid import PID
from longitudinal.gain_scheduler import Gains

class Longitudinal:
    def __init__(self, gains : Gains):
        self.gains = gains
        self.pid = PID(self.gains.KP, self.gains.KI, self.gains.KD)

    def update_des_accel(self, v, v_desired, t):
        return self.pid.update_des_accel(v, v_desired, t)
