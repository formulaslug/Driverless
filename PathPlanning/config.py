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
MAX_TREE_DEPTH = 15   # Maximum number of waypoints to look ahead
K_START = 3           # Number of nearest waypoints to start path tree from

# Path Smoothing Parameters
SPLINE_DEGREE = 3              # Cubic spline
SPLINE_SMOOTHING_FACTOR = 0.1  # Lower = smoother, higher = closer to waypoints
SPLINE_NUM_POINTS = 100        # Number of points in smoothed output path

# Validation Parameters
MIN_CONES_FOR_VALID_PATH = 4  # Minimum cones needed to plan a path

# Cone Labelling
CONE_COLOR_UNKNOWN = -1       # Label for cones whose best color is below MIN_COLOR_CONFIDENCE
MIN_COLOR_CONFIDENCE = 0.4    # Below this the color is treated as unknown (uniform probabilities are 0.25)
ORANGE_NEIGHBOR_RADIUS = 6.0  # Orange cones take the side of the nearest blue/yellow cone within this distance (m)

# Cost Function Parameters
MAX_CONE_TO_PATH_DISTANCE = 5.0  # Cones farther than this from a path are ignored when scoring it (m)
