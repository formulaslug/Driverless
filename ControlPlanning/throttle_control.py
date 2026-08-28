import numpy as np
from config import M, R_WHEEL, T_GEARBOX, C_DRAG, P_AIR, A_FRONT, C_R1, J_E

def torque_load(c_drag, p_air, A_front, c_r1, r_eff, GR,  v):
    F_g = 0 # flat ground, no slope
    F_drag = 0.5 * c_drag * p_air * A_front * v ** 2
    F_roll = c_r1 * v
    F_load = F_g + F_drag + F_roll

    T_load = F_load * r_eff * GR
    return T_load


class ThrottleControl():
    def __init__(self, J_e = J_E, r_eff = R_WHEEL, GR = T_GEARBOX, 
                c_drag = C_DRAG, p_air = P_AIR, A_front = A_FRONT, c_r1 = C_R1):

        self.J_e = J_e
        self.r_eff = r_eff
        self.GR = GR

        self.c_drag = c_drag
        self.p_air = p_air
        self.A_front = A_front
        self.c_r1 = c_r1


    def accel_to_throttle_brake(self, a_des, v): # velocity unused for now
        T_load = torque_load(self.c_drag, self.p_air, self.A_front, self.c_r1, self.r_eff, self.GR, v)

        T_engine = (self.J_e/ (self.r_eff * self.GR)) * a_des + T_load

        if T_engine >= 0:
            brake_output = 0.0
            throttle_output = T_engine
        else:
            brake_output = -T_engine
            throttle_output = 0.0
        return throttle_output, brake_output