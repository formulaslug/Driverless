from dataclasses import dataclass

@dataclass
class Gains:
    """Class for keeping track of Gains"""
    KP:  float = 0.0
    KI:  float = 0.0
    KD:  float = 0.0
    KAW: float = 0.0
    # velocity: float = 0.0


class GainScheduler:
    def __init__(self, our_gains: Gains, t):
        self.our_gains = our_gains

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

    def start_gain_change(self, target: Gains, t, time_to_change):
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

    def change_gains(self, t):
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

    def print_gains(self):
        print(f"KP: {self.our_gains.KP}\nKI: {self.our_gains.KI}\nKD: {self.our_gains.KD}\nKAW: {self.our_gains.KAW}\n")

    def gain_schedule(self, sm_in, t):
        """
        Automated Gain Scheduling. 
        Changes our Gain Constants based state machine input.

        Input: sm_in = state machien input
               t = current time
        """
        # Gain Profiles based on input from State Machine
        if sm_in == 0:
            if (self.target != self.PROFILES[0]):
                self.start_gain_change(self.PROFILES[0], t, 5.0)
            self.change_gains(t)
        elif sm_in == 1:
            if (self.target != self.PROFILES[1]):
                self.start_gain_change(self.PROFILES[1], t, 5.0)
            self.change_gains(t)
        elif sm_in == 2:
            if (self.target != self.PROFILES[2]):
                self.start_gain_change(self.PROFILES[2], t, 5.0)
            self.change_gains(t)
        elif sm_in == 3:
            if (self.target != self.PROFILES[3]):
                self.start_gain_change(self.PROFILES[3], t, 1.0)
            self.change_gains(t)