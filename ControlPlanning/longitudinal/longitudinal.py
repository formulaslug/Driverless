from pid import PID

class Longitudinal:
    def __init__(self, PID, Gains):
        self.pid = PID(K_p, K_i, K_d)

    def update(self, v, v_desired):
        a_des = self.pid.calc_des_accel(v, v_desired, t)
        return a_des