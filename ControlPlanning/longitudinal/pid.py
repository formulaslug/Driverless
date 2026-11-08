from dataclasses import dataclass
import numpy as np
from config import MAX_ACCEL, MAX_BRAKE

@dataclass
class Gains:
    """Class for keeping track of Gains"""
    KP:  float = 0.0
    KI:  float = 0.0
    KD:  float = 0.0
    KAW: float = 0.0
    velocity: float = 0.0

class PID:

    def __init__(self, K_p, K_i=0.0, K_d=0.0, K_aw=0.0):
        self.t_prev : float = 0.0
        self.v : float = 0.0
        self.v_prev : float = 0.0
        self.a_des_prev : float = 0.0
        self.error_prev : float = 0.0
        self.net_integral : float = 0.0
        self.intialized : bool = False
        self.gains = Gains(K_p, K_i, K_d, K_aw)

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
        self.rate = Gains(0.0, 0.0, 0.0, 0.0)

        self.PROFILES = {
            0: Gains(0.0, 0.0, 0.0, 0.0),
            1: Gains(1.0, 1.0, 1.0, 1.0)
        }

        self.dkp : float = 0.0
        self.dki : float = 0.0
        self.kd  : float = 0.0
        self.dkaw : float = 0.0

        self.t_ramp_end : float = 0.0

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
    
    def start_gain_change(self, target: Gains, t, time_to_change):
        """
        Initializes values for a Gain Profile transition
        """
        self.target = target
        self.t_ramp_end = t + time_to_change
        g = self.gains

        # defining our rage of change per gain
        self.rate = Gains(
            (target.KP - g.KP) / time_to_change,
            (target.KI - g.KI) / time_to_change,
            (target.KD - g.KD) / time_to_change,
            (target.KAW - g.KAW) / time_to_change
        )

    def change_gains(self, t, dt):
        """
        Starts changing our gains gradually over time, transient-free
        """
        if self.target is None:
            return
        
        # Change Gains to target if time is up
        if t >= self.t_ramp_end:
            self.gains = self.target
            self.target = None
            return
        
        # Gradually change gains while we are under our time limit
        g = self.gains
        r = self.rate
        self.gains = Gains(
            g.KP + r.KP * dt,
            g.KI + r.KI * dt,
            g.KD + r.KD * dt,
            g.KAW + r.KAW * dt
        )

    def print_gains(self):
        print(f"KP: {self.gains.KP}\nKI: {self.gains.KI}\nKD: {self.gains.KD}\nKAW: {self.gains.KAW}\n")
    
    def gain_schedule(self, sm_in, t, dt):
        # Gain Profiles based on input from State Machine
        if sm_in == 0:
            if (self.target is None):
                self.start_gain_change(self.PROFILES[0], t, 5.0)
            self.change_gains(t, dt)
        elif sm_in == 1:
            if (self.target is None):
                self.start_gain_change(self.PROFILES[1], t, 5.0)
            self.change_gains(t, dt)

    def update_des_accel(self, v, v_desired, t):
        """
        PID longitudinal algorithm, updating state variables, and desired accel
        """
	    # compute feedback
        error = v_desired - v
        if not self.intialized:
            dt = 0.01
            self.v_prev = v
            self.intialized = True
        elif t - self.t_prev < 0.001:
            return self.a_des_prev
        else:
            dt = t - self.t_prev

        P = error * self.gains.KP
        D = -(v - self.v_prev) / dt * self.gains.KD # prevent deritive kick
        # D = (error - self.error_prev) / dt * self.gains.KD

        # raw_output = self.K_p * P + self.K_i * I + self.K_d * D

        # Integral Clamp
        I_check = self.gains.KI * (self.net_integral + error * dt)
        check_output = P + I_check + D
        max_output = MAX_ACCEL
        min_output = MAX_BRAKE

        saturated_high = check_output >= max_output and error > 0
        saturated_low = check_output <= min_output and error < 0

        if not (saturated_high or saturated_low):
            self.net_integral += error * dt

        I = self.gains.KI * self.net_integral

        # PID raw output
        a_des = P + I + D
        a_des = np.clip(a_des, min_output, max_output)
        # self.gain_schedule(sm_in=0, t=t, dt=dt)

        # Find Rise Time
        if (not self.rise_time_defined):
            self.find_rise_time(t=t, v=v, v_desired=v_desired)

        # Find Settle Time
        self.v_upper_bound = v_desired * 1.01
        self.v_lower_bound = v_desired * 0.99
        self.find_settle_time(v=v, t=t)

        # Update state variables
        self.t_prev = t
        self.v_prev = v
        self.error_prev = error
        self.a_des_prev = a_des
        return a_des 


    