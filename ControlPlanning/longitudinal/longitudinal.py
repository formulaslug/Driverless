from pid import PID

class Longitudinal:
    def __init__(self, K_p, K_i, K_d, Gains = None, v_des = 0.0):
        if Gains is not None:
            K_p = Gains.KP
            K_i = Gains.KI
            K_d = Gains.KD
        self.v_des = v_des
        self.pid = PID(K_p, K_i, K_d)

    def update_des_accel(self, v, v_desired, t):
        return self.pid.update_des_accel(v, v_desired, t)
