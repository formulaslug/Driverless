#Make up fake poses & cones for testing transformations

#1) Creates a few car poses along a rectangle (global frame).
#2) Creates some cones around the track in global coordinates
#3) For each pose, converts cone positions into vehicle frame.
#4) Then converts those vehicle-frame cone positions back to global frame as your “relative→global” mapping.
#5) Stores everything in Polars DataFrames 


import math
import numpy as np
import polars as pl

from localization.se2_utils import se2_from_xytheta, transform_points, invert_se2
