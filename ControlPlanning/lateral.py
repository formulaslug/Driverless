import numpy as np

class Lateral():
    def __init__(self, K_dd, min_lookahead, wheelbase):
        self.K_dd = K_dd
        self.min_lookahead = min_lookahead
        self.wheelbase = wheelbase



    def distance_to(self, tar_waypoint, x, y):
        """
        calculates distance to a target waypoint from current car 2d coords
        """
        tar_x, tar_y, tar_v = tar_waypoint
        return np.sqrt((tar_x - x) ** 2  + (tar_y - y) ** 2 )

    def get_curve(self, alpha = 0, l_d = 0, R = 0):
        """
        Retries curvature using alpha(angle from vehichle heading to lookahead vector), and the lookahead vector distance.
        Also able to pass in radius of curve as R
        """
        if R != 0:
            return 1 / R
        else:
            return 2 * np.sin(alpha) / l_d

    def update(self, waypoints, x, y, yaw, v)
        """
        
        """
        lookahead = list()
        min_lookahead = 3.0
        L = max(self.K_dd * v, min_lookahead)


        closest_idx = 0
        min_dist = float('inf')
        for i in range(len(waypoints)):
            dist = self.distance_to(waypoints[i], x, y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i


        lookahead = waypoints[-1]  # fallback
        for waypoint in waypoints[closest_idx:]:
            if distance_to(waypoint) >= L:
                lookahead = list(waypoint)
                break
            
        l_d = distance_to(lookahead)
        lookahead_vector = [lookahead[0] - x, lookahead[1] - y]
        heading_vector = [np.cos(yaw), np.sin(yaw)]

        alpha = np.arctan2(lookahead_vector[1], lookahead_vector[0]) - (yaw)
        k = get_curve(alpha=alpha, l_d=l_d)

        delta = np.arctan(k * 1.5)