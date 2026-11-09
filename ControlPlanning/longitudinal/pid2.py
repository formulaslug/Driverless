from dataclasses import dataclass
import numpy as np
import gain_scheduler as gs

# from gain_scheduler import Gains
# from gain_scheduler import start_gain_change
# from gain_scheduler import change_gains
# from gain_scheduler import print_gains
# from gain_scheduler import gain_schedule

class PID:
    def __init__(self, K_p, K_i=0.0, K_d=0.0, K_aw=0.0):
        self.t_prev : float = 0.0
        self.v : float = 0.0
        self.v_prev : float = 0.0
        self.error_prev : float = 0.0
        self.net_integral : float = 0.0
        self.intialized : bool = False
        self.gains = gs.Gains(K_p, K_i, K_d, K_aw)

        ### Variables used to determine rise time
        self.t_10 : float = None
        self.t_90 : float = None
        self.rise_time : float = 0.0
        self.rise_time_defined : bool = False

        ### Variables used to determine settle time
        self.settle_time : float = None
        self.in_bound : bool = False

        # Error Band Vars
        self.v_lower_bound : float = None
        self.v_upper_bound : float = None

        ### Variables used for Gain Scheduling
        self.target = None
        self.rate = gs.Gains(0.0, 0.0, 0.0, 0.0)

        self.PROFILES = {
            0: gs.Gains(0.0, 0.0, 0.0, 0.0),
            1: gs.Gains(1.0, 1.0, 1.0, 1.0)
        }

        self.dkp : float = 0.0
        self.dki : float = 0.0
        self.kd  : float = 0.0
        self.dkaw : float = 0.0

        self.t_ramp_end : float = 0.0


    def calc_possible_output(self):
        """
        calculates the actual possible output for the car, not raw pid
        """
        pass

    def find_rise_time(self, t, v, v_desired):
        """
        Finds the rise time of our car.
        Calculates the time differenec between time we reach 90% of v_desired to 10% of v_desired.
        """
        # We must calculate rise time from 10% to 90% of our target velocity
        if (self.t_10 is None and v >= 0.1 * v_desired):
                self.t_10 = t
            
        if (self.t_90 is None and v >= 0.9 * v_desired):
            self.t_90 = t
            self.rise_time = self.t_90 - self.t_10
            self.rise_time_defined = True
    
    def find_settle_time(self, v, t):
        """
        Finds settle time of car. Records time at which we enter and reside in our Error Band.
        """
        self.in_bound = v >= self.v_lower_bound and v <= self.v_upper_bound
        if self.in_bound:
            if self.settle_time is None:
                self.settle_time = t
        else:
            self.settle_time = None


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

        # raw_output = self.K_p * P + self.K_i * I + self.K_d * D
        raw_output = self.gains.KP * P + self.gains.KI * I + self.gains.KD * D

        gs.gain_schedule(sm_in=0, t=t)

        # add backcalculation for integral windup here
        # possible_output = self.calc_possible_output() #calculates the possible output
        # back_calculation = self.K_aw * (possible_output - raw_output) #back calculation for integral windup
        # self.net_integral -= dt * self.K_i * back_calculation #back calculation for integral windup

        # Find Rise Time
        if (not self.rise_time_defined):
            self.find_rise_time(t=t, v=v, v_desired=v_desired)

        # Find Settle Time
        self.v_upper_bound = v_desired * 1.01
        self.v_lower_bound = v_desired * 0.99
        self.find_settle_time(v=v, t=t)

        return raw_output
        # return a_des 


    
