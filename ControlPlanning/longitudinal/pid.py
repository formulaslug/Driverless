from dataclasses import dataclass
import numpy as np
from gain_scheduler import Gains, GainScheduler
from config import MAX_ACCEL, MAX_BRAKE

class PID:

    def __init__(self, K_p, K_i=0.0, K_d=0.0, K_aw=0.0):
        self.t_prev : float = 0.0
        self.v : float = 0.0
        self.v_prev : float = 0.0
        self.a_des_prev : float = 0.0
        self.error_prev : float = 0.0
        self.net_integral : float = 0.0
        self.intialized : bool = False

        # Initialize GainScheduler with a Gains object
        intial_gains = Gains(K_p, K_i, K_d, K_aw)
        self.gains = GainScheduler(intial_gains, t=0.0)

        # Variables used to determine rise time
        self.t_10 : float = None
        self.t_90 : float = None
        self.rise_time : float = 0.0
        self.rise_time_defined : bool = False

        # Variables used to determine settle time
        self.settle_time : float = None
        self.in_bound : bool = False

        # Vairbles used to determine overshoot
        self.overshoot : float = 0.0
        self.v_max : float = 0.0

        # Error Band Vars
        self.v_lower_bound : float = None
        self.v_upper_bound : float = None

    def find_rise_time(self, t, v, v_desired):
        """
        Finds the rise time of our car.
        Calculates the time differenec between time we reach 90% of v_desired to 10% of v_desired.
        """
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

    def update_des_accel(self, v, v_desired, t, sm_in=0):
        """
        PID longitudinal algorithm, updating state variables, and desired accel
        """
	    # Compute feedback error
        error = v_desired - v
        if not self.intialized:
            dt = 0.01
            self.v_prev = v
            self.intialized = True
        elif t - self.t_prev < 0.001:
            return self.a_des_prev
        else:
            dt = t - self.t_prev

        # Update gains via GainScheduler before computing PID
        self.gains.gain_schedule(sm_in, t=t)

        P = error * self.gains.our_gains.KP
        D = -(v - self.v_prev) / dt * self.gains.our_gains.KD # prevent deritive kick

        # Integral Clamp
        I_check = self.gains.our_gains.KI * (self.net_integral + error * dt)
        check_output = P + I_check + D
        max_output = MAX_ACCEL
        min_output = MAX_BRAKE

        saturated_high = check_output >= max_output and error > 0
        saturated_low = check_output <= min_output and error < 0

        if not (saturated_high or saturated_low):
            self.net_integral += error * dt

        I = self.gains.our_gains.KI * self.net_integral

        # PID raw output
        a_des = P + I + D
        a_des = np.clip(a_des, min_output, max_output)

        # Find Rise Time
        if (not self.rise_time_defined):
            self.find_rise_time(t=t, v=v, v_desired=v_desired)

        # Find Settle Time
        self.v_upper_bound = v_desired * 1.01
        self.v_lower_bound = v_desired * 0.99
        self.find_settle_time(v=v, t=t)

        # Find overshoot
        if v > self.v_max:
            self.v_max = v
        if v_desired > 0:
            self.overshoot = max(0.0, (self.v_max - v_desired) / v_desired * 100.0)

        # Update state variables
        self.t_prev = t
        self.v_prev = v
        self.error_prev = error
        self.a_des_prev = a_des
        return a_des 
