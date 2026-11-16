
from lateral.pure_pursuit import PurePursuit
from lateral.stanley import Stanley
from config import K_DD, MIN_LOOKAHEAD, WHEELBASE

class Lateral():
    def __init__(self):
        self.pure_pursuit = PurePursuit(K_dd=K_DD, min_lookahead=MIN_LOOKAHEAD, wheelbase=WHEELBASE)

    def update(self, waypoints, x, y, yaw, v):
        return self.pure_pursuit.update_delta(waypoints, x, y, yaw, v)