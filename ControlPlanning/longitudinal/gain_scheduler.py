from dataclasses import dataclass

@dataclass
class Gains:
    """Class for keeping track of Gains"""
    KP:  float = 0.0
    KI:  float = 0.0
    KD:  float = 0.0


class GainScheduler:
    def __init__(self, initial_gains: Gains, t: float = 0.0):
        self.gains = initial_gains
        self.t_prev = t
        self.target : Gains = None
        self.rate : Gains = None
        self.t_ramp_end : float = None

        # Stored in tuple(min desired velocity, gains, ramp time)
        # Append new profiles, indexed by order of list
        self.PROFILES = [
            (0.0, initial_gains, 2.0)
        ]


    def start_gain_change(self, target: Gains, t: float, time_to_change: float):
            """
            Initializes values for a Gain Profile transition.
            
            Input: target : Gains = a Gain Profile we are trying to reach
                   t = current time
                   time_to_change = the time in which we want to reach our target
            """
            self.target = target
            self.t_prev = t
            self.t_ramp_end = t + time_to_change
            g = self.gains

            # defining our rate of change per gain
            self.rate = Gains(
                (target.KP - g.KP) / time_to_change,
                (target.KI - g.KI) / time_to_change,
                (target.KD - g.KD) / time_to_change,
            )

    def change_gains(self, t: float):
        """
        Starts changing our gains gradually over time, transient-free.
        
        Input: t = current time
        """
        dt = t - self.t_prev 
        if self.target is None:
            return
        
        # Change Gains to target if time is up
        if (self.gains == self.target):
            self.target = None
            return
        elif t >= self.t_ramp_end:
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
        )

        self.t_prev = t


    def gain_schedule(self, v_desired: float, t: float):
        """
        Automated Gain Scheduling. 
        Changes our Gain Constants based on desired velocity.

        Input: v_desired = desired velocity (m/s)
               t = current time (s)
        """
        # Set initial gains
        min_des_v, desired_profile, ramp_time = self.PROFILES[0]
        
        # Iterate through to check which gains should be active
        for min_des_v, gains, ramp in self.PROFILES:
            if v_desired >= min_des_v:
                desired_profile, ramp_time = gains, ramp

    
        # Only start a new ramp if we aren't already targeting it AND aren't already at it
        if self.target != desired_profile and self.gains != desired_profile:
            self.start_gain_change(desired_profile, t, ramp_time)
            
        self.change_gains(t)

    def print_gains(self):
        print(f"KP: {self.gains.KP}\nKI: {self.gains.KI}\nKD: {self.gains.KD}\n")