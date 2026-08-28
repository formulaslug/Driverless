class Longitudinal:
    def __init__(self, K_p, K_i, K_d):
        self.t_previous = 0.0
        self.v_previous = 0.0
        self.error_previous = 0.0
        self.net_integral = 0.0
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
    def update(self, v, v_desired, t):
        """
        PID longitudinal algorithm, updating state variables, and desired accel
        """

        # PID algo to compute desired accel
        if self.t_previous == 0.0:
             dt = 0.01

        else:
             dt = t - self.t_previous
        self.net_integral +=  dt * (v_desired - v)
        P = v_desired - v
        I = self.net_integral
        D = ( (v_desired - v) - self.error_previous ) / dt
        a_des = self.K_p * P + self.K_i * I + self.K_d * D


        self.error_previous = v_desired - v
        self.t_previous = t

        return a_des


    
