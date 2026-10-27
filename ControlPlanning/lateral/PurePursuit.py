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
        tar_x, tar_y= tar_waypoint
        return np.sqrt((tar_x - x) ** 2  + (tar_y - y) ** 2 )

    def get_curve(self, alpha = 0, l_d = 0, R = 0):
        """
        Retries curvature using alpha(angle from vehichle heading to lookahead vector), and the lookahead vector distance.
        Also able to pass in radius of curve as R
        """
        if l_d == 0:
            return 0
        if R != 0:
            return 1 / R
        else:
            return 2 * np.sin(alpha) / l_d

    def update(self, waypoints, x, y, yaw, v):
        """
        pure pursuit algorithm, uses lookahead point from set of waypoinys which dist scales with velocity,
        and returns a steering angle in radians(delta)
        """
        #L scales on velocity fo k_dd constant, min_lookahead is a safety if v is near 0
        L = max(self.K_dd * v, self.min_lookahead)


        closest_idx = 0
        min_dist = float('inf')
        for i in range(len(waypoints)):
            dist = self.distance_to(waypoints[i], x, y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        lookahead = waypoints[-1]  # safety to end waypoint 

        #for each waypoint past the index of the closest one, set the one to lookahead if its past distance L
        for waypoint in waypoints[closest_idx:]:
            if self.distance_to(waypoint, x, y) >= L:
                lookahead = list(waypoint)
                break
            
        l_d = self.distance_to(lookahead, x, y)
        lookahead_vector = [lookahead[0] - x, lookahead[1] - y]

        alpha = np.arctan2(lookahead_vector[1], lookahead_vector[0]) - (yaw)
        k = self.get_curve(alpha=alpha, l_d=l_d)

        delta = np.arctan(k * self.wheelbase)

        return delta #delta is front wheel turning angleso 