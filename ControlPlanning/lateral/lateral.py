
from lateral.pure_pursuit import PurePursuit
from lateral.stanley import Stanley

class Lateral():
    def __init__(self, K_dd=1.0, min_lookahead=2.0, wheelbase=2.5):
        self.pure_pursuit = PurePursuit(K_dd=K_dd, min_lookahead=min_lookahead, wheelbase=wheelbase)
        self.stanley = Stanley(k=1.0, wheelbase=wheelbase)

    def update(self, waypoints, x, y, yaw, v):
        return self.pure_pursuit.update_curvature(waypoints, x, y, yaw, v)