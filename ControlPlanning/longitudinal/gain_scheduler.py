from dataclasses import dataclass

@dataclass
class Gains:
    """Class for keeping track of Gains"""
    KP:  float = 0.0
    KI:  float = 0.0
    KD:  float = 0.0
    KAW: float = 0.0


class GainScheduler:
    def __init__(self, initial_gains: Gains, t: float = 0.0):
        self.our_gains = initial_gains
        self.t_prev = t
        self.target : Gains = None
        self.rate : Gains = None
        self.t_ramp_end : float = None

        self.PROFILES = {
            0: Gains(0.0, 0.0, 0.0, 0.0),
            1: Gains(1.0, 1.0, 1.0, 1.0),
            2: Gains(10.0, 10.0, 10.0, 10.0),
            3: Gains(1.5, 7.6, 4.9, 0.15)
        }

        self.RAMP_TIMES = {
            0: 5.0,
            1: 4.0,
            2: 3.0,
            3: 2.0
        }

    def start_gain_change(self, target: Gains, t: float, time_to_change: float):
            """
            Initializes values for a Gain Profile transition.
            
            Input: target : Gains = a Gain Profile we are trying to reach
                   t = current time
                   time_to_change = the time in which we want to reach our target
            """
            self.target = target
            self.t_ramp_end = t + time_to_change
            g = self.our_gains

            # defining our rate of change per gain
            self.rate = Gains(
                (target.KP - g.KP) / time_to_change,
                (target.KI - g.KI) / time_to_change,
                (target.KD - g.KD) / time_to_change,
                (target.KAW - g.KAW) / time_to_change
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
        if (self.our_gains == self.target):
            self.target = None
            return
        elif t >= self.t_ramp_end:
            self.our_gains = self.target
            self.target = None
            return
        
        # Gradually change gains while we are under our time limit
        g = self.our_gains
        r = self.rate
        self.our_gains = Gains(
            g.KP + r.KP * dt,
            g.KI + r.KI * dt,
            g.KD + r.KD * dt,
            g.KAW + r.KAW * dt
        )

        self.t_prev = t

    def gain_schedule(self, sm_in: int, t: float):
        """
        Automated Gain Scheduling. 
        Changes our Gain Constants based state machine input.

        Input: sm_in = state machine input
               t = current time
        """
        if sm_in in self.PROFILES:
            desired_profile = self.PROFILES[sm_in]
            ramp_time = self.RAMP_TIMES[sm_in]

            # Only start a new ramp if we aren't already targeting it AND aren't already at it
            if self.target != desired_profile and self.our_gains != desired_profile:
                self.start_gain_change(desired_profile, t, ramp_time)
            
            self.change_gains(t)

    def print_gains(self):
        print(f"KP: {self.our_gains.KP}\nKI: {self.our_gains.KI}\nKD: {self.our_gains.KD}\nKAW: {self.our_gains.KAW}\n")