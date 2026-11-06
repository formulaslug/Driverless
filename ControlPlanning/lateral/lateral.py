

from pure_pursuit import PurePursuit
from stanley import Stanley

class Lateral():
    def __init__(self):
        self.pure_pursuit = PurePursuit(K_dd=1.0, min_lookahead=2.0, wheelbase=2.5)
        self.stanley = Stanley(k=1.0, wheelbase=2.5)
        
    def update(self, waypoints, x, y, yaw, v):
        return self.pure_pursuit.get_curvature(waypoints, x, y, yaw, v)