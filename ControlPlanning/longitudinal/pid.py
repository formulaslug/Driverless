from dataclasses import dataclass
import numpy as np

class PID:
    def __init__(self, K_p, K_i=0.0, K_d=0.0, K_aw=0.0):
        self.t_prev : float = 0.0
        self.v_prev : float = 0.0
        self.error_prev : float = 0.0

        self.K_p : float = K_p
        self.K_i : float = K_i
        self.K_d : float = K_d
        self.K_aw : float = K_aw
        self.net_integral : float = 0.0

        self.intialized : bool = False

    def calc_possible_output(self):
        """
        calculates the actual possible output for the car, not raw pid
        """
        pass
    def update(self, v, v_desired, t):
        """
        PID longitudinal algorithm, updating state variables, and desired accel
        """
	# compute feedback
        error = v_desired - v
        if not self.intialized:
            dt = 0.01
            self.intialized = True
        else:
             dt = t - self.t_prev

        self.net_integral += error * dt

        P = error
        I = self.net_integral
        D = (error - self.error_prev) / dt
        raw_output = self.K_p * P + self.K_i * I + self.K_d * D

        #add backcalculation for integral windup here
        possible_output = self.calc_possible_output() #calculates the possible output
        back_calculation = self.K_aw * (possible_output - raw_output) #back calculation for integral windup
        self.net_integral -= dt * self.K_i * back_calculation #back calculation for integral windup

        return a_des 


    
