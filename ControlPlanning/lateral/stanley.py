

class Stanley():
    def __init__(self, k, wheelbase):
        self.k = k
        self.wheelbase = wheelbase

    def get_curvature(self, waypoints, x, y, yaw, v):
        return 0.0  