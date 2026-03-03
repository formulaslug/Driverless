"""
Path Planning Configuration
Contains all constants and parameters for the path planning system
"""

# Cone Colors (indices into color probability array)
CONE_COLOR_BLUE = 0          # Left track border
CONE_COLOR_YELLOW = 1        # Right track border
CONE_COLOR_ORANGE_SMALL = 2  # Entry/exit lanes
CONE_COLOR_ORANGE_LARGE = 3  # Start/finish lines

# Path Tree & Beam Search Parameters
BEAM_WIDTH = 10       # Number of best paths to keep at each tree level
MAX_TREE_DEPTH = 30   # Maximum number of waypoints to look ahead
K_START = 3           # Number of nearest waypoints to start path tree from

# Path Smoothing Parameters
SPLINE_DEGREE = 3              # Cubic spline
SPLINE_SMOOTHING_FACTOR = 0.1  # Lower = smoother, higher = closer to waypoints
SPLINE_NUM_POINTS = 100        # Number of points in smoothed output path

# Validation Parameters
MIN_CONES_FOR_VALID_PATH = 4  # Minimum cones needed to plan a path
