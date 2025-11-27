"""
Path Planning Configuration
Contains all constants and parameters for the path planning system
"""

# Performance Requirements
PLANNING_FREQUENCY_HZ = 25  # Hz - must compute new path 25 times/sec
CYCLE_TIME_MS = 40  # ms - max computation time (1000/25)

# Track Constraints (from Formula Student Driverless rules)
MIN_TRACK_WIDTH = 3.0  # meters (DD.2.2.2.c)
MIN_TURNING_DIAMETER = 9.0  # meters (DD.2.2.2.d)
MIN_TURNING_RADIUS = MIN_TURNING_DIAMETER / 2  # 4.5 meters
MAX_STRAIGHT_LENGTH = 80.0  # meters (DD.2.2.2.a)
LAP_LENGTH_MIN = 200.0  # meters (DD.2.2.2.e)
LAP_LENGTH_MAX = 500.0  # meters (DD.2.2.2.e)

# Cone Colors (from DD.1.1)
CONE_COLOR_BLUE = 0  # Left track border
CONE_COLOR_YELLOW = 1  # Right track border
CONE_COLOR_ORANGE_SMALL = 2  # Entry/exit lanes
CONE_COLOR_ORANGE_LARGE = 3  # Start/finish lines

# Beam Search Parameters
BEAM_WIDTH = 10  # Number of best paths to keep at each tree level
MAX_TREE_DEPTH = 15  # Maximum number of waypoints to look ahead

# Cost Function Weights
WEIGHT_BOUNDARY_VIOLATION = 1000.0  # Heavily penalize leaving track
WEIGHT_TURNING_RADIUS = 100.0  # Penalize sharp turns
WEIGHT_CURVATURE_CHANGE = 50.0  # Prefer smooth paths
WEIGHT_DISTANCE = 1.0  # Prefer shorter paths
WEIGHT_CENTER_LINE = 10.0  # Prefer staying near center of track

# Path Smoothing Parameters
SPLINE_DEGREE = 3  # Cubic spline
SPLINE_SMOOTHING_FACTOR = 0.1  # Lower = smoother, higher = closer to waypoints

# Vehicle Dynamics (estimated - coordinate with Control Planning team)
VEHICLE_MAX_SPEED = 20.0  # m/s (72 km/h - from competitor data)
VEHICLE_MAX_ACCELERATION = 15.0  # m/s^2 (0-100 km/h in ~2s)

# Noise/Robustness Parameters
MAX_CONE_POSITION_ERROR = 0.5  # meters - expected SLAM noise from monocular vision
MIN_CONES_FOR_VALID_PATH = 4  # Minimum cones needed to plan a path
