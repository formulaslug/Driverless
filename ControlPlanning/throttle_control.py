from config import R_WHEEL, T_GEARBOX, C_DRAG, P_AIR, A_FRONT, C_R1, J_E 
import numpy as np

class ThrottleControl():
    def __init__(self, J_E = J_E, r_eff = R_WHEEL, GR = T_GEARBOX, 
                c_drag = C_DRAG, p_air = P_AIR, A_front = A_FRONT, c_r1 = C_R1):

        self.J_E = J_E
        self.r_eff = r_eff
        self.GR = GR

        self.c_drag = c_drag
        self.p_air = p_air
        self.A_front = A_front
        self.c_r1 = c_r1

    
    def torque_load(self, v):
        F_g = 0 # flat ground, no gravity load on horizontal plane
        F_drag = 0.5 * self.c_drag * self.p_air * self.A_front * v ** 2
        F_roll = self.c_r1 * v
        F_load = F_g + F_drag + F_roll

        T_load = F_load * self.r_eff * self.GR
        return T_load
    def accel_to_throttle_brake(self, a_des, v):
        T_load = self.torque_load(v)

        T_motor = (self.J_E/ (self.r_eff * self.GR)) * a_des + T_load

        if T_motor >= 0:
            brake_output = 0.0
            throttle_output = np.clip(T_motor, 0, 1)
        else:
            brake_output = -np.clip(T_motor, -1, 0)
            throttle_output = 0.0
        return throttle_output, brake_output