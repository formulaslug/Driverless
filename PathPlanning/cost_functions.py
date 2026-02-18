
# heuristic cost function for evaluating best candidate path based on 5 metrics (AMZ)
#    1. angle sharpness   - penalizes big/sudden turns unless necessary 
#    2. smooth curves - penalizes randomized direction changes
#    3. cone spacing - cones on each boundary are usually evenly spaced
#    4. color confidence - penalizes cones whose color classification is uncertain or wrong 
#    5. path length - prefers paths witha 12m range can be changeable 
# for refrence 1 and 2 are combined in angle + curve metric

import numpy as np
import config as cfg
from scipy.spatial.distance import pdist, squareform

TARGET_PATH_LENGTH = 15.0   # desired horizon
BIG_COST = 1e9

def assign_cones_to_segments(path, cones):
    n_cones = len(cones)
    n_segments = len(path) - 1
    
    # create np arrays for the distances and cross products of each cone to each segment
    distances = np.zeros((n_cones, n_segments))
    cross_products = np.zeros((n_cones, n_segments))
    
    for idx in range(n_segments):
        A = path[idx]   # first segment waypoint
        B = path[idx + 1]   # second segment waypoint
        vec_AB = B - A
        seg_len_sq = np.dot(vec_AB, vec_AB) # squared length of the segment
        
        if seg_len_sq < 1e-10:  # segment length too small
            distances[:, idx] = np.inf
            cross_products[:, idx] = 0
            continue
        
        vec_AC = cones - A # shape: (n_cones, 2)
        t = np.dot(vec_AC, vec_AB) / seg_len_sq
        
        # only keep cones with t values between [0,1]
        valid_mask = (t>=0) & (t<=1) # boolean array
        
        # initialize with large distances to filter out invalid cones
        distances[:, idx] = np.inf
        cross_products[:, idx] = 0
        
        if np.any(valid_mask):
            valid_t = t[valid_mask]
            valid_cones = cones[valid_mask]
            valid_AC = vec_AC[valid_mask]
            
            # calculates the closest point on the segment for each cone
            projections = A + valid_t[:, np.newaxis] * vec_AB
            # calculate distances between each cone and its projection point
            distances[valid_mask, idx] = np.linalg.norm(valid_cones - projections, axis=1)
            # calculate cross product to find which side the cone is on
            cross_products[valid_mask, idx] = vec_AB[0] * valid_AC[:, 1] - vec_AB[1] * valid_AC[:, 0]
        
    # assign each cone to its closest segment
    segment_assignments = np.argmin(distances, axis=1)
    # create empty array to assign sides to each cone
    side_assignments = np.zeros(n_cones, dtype=int)
    
    for i in range(n_cones):
        seg = segment_assignments[i]
        cross = cross_products[i, seg]
        
        if cross > 1e-6:
            side_assignments[i] = 1 # cone is on the left side of the segment
        elif cross < -1e-6:
            side_assignments[i] = -1 # cone is on the right side of the segment
        else:
            side_assignments[i] = 0 # cone could be on the line
    
    return segment_assignments, side_assignments
        
def count_boundary_violations(cone_colors, side_assignments):
    # filter out orange cones and cones that do not have a left or right side assignment
    valid_mask = (cone_colors != cfg.CONE_COLOR_ORANGE_SMALL) & \
                    (cone_colors != cfg.CONE_COLOR_ORANGE_LARGE) & \
                    (side_assignments != 0)
                    
    expected_sides = np.zeros_like(cone_colors, dtype=int)
    expected_sides[cone_colors == cfg.CONE_COLOR_BLUE] = 1 # left side cones should be blue
    expected_sides[cone_colors == cfg.CONE_COLOR_YELLOW] = -1 # right side cones should be yellow
    
    mismatches = (expected_sides != side_assignments) & valid_mask
    return int(np.sum(mismatches))

def calculate_boundary_violation(path, cones, colors):   
    cone_colors = np.argmax(colors, axis=1)
    segment_assignments, side_assignments = assign_cones_to_segments(path, cones)
    violations = count_boundary_violations(cone_colors, side_assignments)
    
    non_orange_mask = (cone_colors != cfg.CONE_COLOR_ORANGE_SMALL) & \
                      (cone_colors != cfg.CONE_COLOR_ORANGE_LARGE)
    n_non_orange = np.sum(non_orange_mask)

    if n_non_orange == 0:
        return 0.0

    return float(violations) / float(n_non_orange)

def calculate_trackwidth_variance(path, cones, colors):
    if colors is None or len(colors) == 0:
        return 0.0

    cone_colors = np.argmax(colors, axis=1)
    segment_assignments, side_assignments = assign_cones_to_segments(path, cones)

    n_segments = len(path) - 1
    widths = []

    for seg_idx in range(n_segments):
        # find cones assigned to this segment
        seg_mask = (segment_assignments == seg_idx)

        # separate by side: left (blue, side=1) and right (yellow, side=-1)
        left_mask = seg_mask & (side_assignments == 1) & (cone_colors == cfg.CONE_COLOR_BLUE)
        right_mask = seg_mask & (side_assignments == -1) & (cone_colors == cfg.CONE_COLOR_YELLOW)

        if not (np.any(left_mask) and np.any(right_mask)):
            continue  # Skip segments without both left and right cones

        # get segment vector, same as assign_cones_to_segments function
        A = path[seg_idx]
        B = path[seg_idx + 1]
        vec_AB = B - A
        seg_len_sq = np.dot(vec_AB, vec_AB)

        if seg_len_sq < 1e-10:
            continue

        # calculate perpendicular distances for left and right cones
        left_cones = cones[left_mask]
        right_cones = cones[right_mask]

        # for simplicity, use closest cone on each side
        vec_AC_left = left_cones - A
        vec_AC_right = right_cones - A

        # project onto segment to get perpendicular distances
        t_left = np.dot(vec_AC_left, vec_AB) / seg_len_sq
        t_right = np.dot(vec_AC_right, vec_AB) / seg_len_sq

        # get closest points on segment
        proj_left = A + np.clip(t_left, 0, 1)[:, np.newaxis] * vec_AB
        proj_right = A + np.clip(t_right, 0, 1)[:, np.newaxis] * vec_AB

        # calculate perpendicular distances
        dist_left = np.linalg.norm(left_cones - proj_left, axis=1)
        dist_right = np.linalg.norm(right_cones - proj_right, axis=1)

        # use minimum distance (closest cone) on each side
        min_dist_left = np.min(dist_left)
        min_dist_right = np.min(dist_right)

        # track width is sum of perpendicular distances
        width = min_dist_left + min_dist_right
        widths.append(width)

    if len(widths) < 2:
        return 0.0

    # return variance (or standard deviation) of track widths
    return float(np.var(widths))


def evaluate_path_cost(path, cones, coordinate_confidence, colors=None):
    path = np.asarray(path, float)
    cones = np.asarray(cones, float)

    if len(path) < 2 or len(cones) < cfg.MIN_CONES_FOR_VALID_PATH:
        return BIG_COST

    # angle + curve metric
    segs = path[1:] - path[:-1]
    headings = np.arctan2(segs[:, 1], segs[:, 0])

    if len(headings) > 1:
        dtheta = np.diff(headings)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))  # wrap
        angle_metric = float(np.max(np.abs(dtheta))**2)
        smoothness_metric = float(np.sum(dtheta**2))
    else:
        angle_metric = 0.0
        smoothness_metric = 0.0

    # cone spacing metric (using scipy for efficiency)
    if len(cones) > 1:
        # Compute pairwise distances efficiently
        dist_matrix = squareform(pdist(cones))  # shape: (n_cones, n_cones)
        np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-distances

        # Find nearest neighbor distance for each cone
        nn_dist = np.min(dist_matrix, axis=1)
        spacing_metric = float(np.var(nn_dist))
    else:
        spacing_metric = 0.0

    # color metric
    color_metric = 0.0
    if colors is not None:
        colors = np.asarray(colors, float)
        if colors.shape[0] == cones.shape[0]:
            p_best = np.max(colors, axis=1)   # most likely color
            wrong_p = 1.0 - p_best           # chance of being wrong
            color_metric = float(np.mean(wrong_p)**2)

    # length metric
    seg_lengths = np.linalg.norm(segs, axis=1)
    total_length = float(np.sum(seg_lengths))
    length_metric = (total_length - TARGET_PATH_LENGTH) ** 2

    # track width variance metric
    if colors is not None:
        trackwidth_metric = calculate_trackwidth_variance(path, cones, colors)
    else:
        trackwidth_metric = 0.0

    # track boundary metric
    if colors is not None:
        boundary_violation_metric = calculate_boundary_violation(path, cones, colors)
    else:
        boundary_violation_metric = 0.0
        
    # coordinate confidence metric — penalize paths near positionally uncertain cones
    coord_conf_metric = 0.0
    if coordinate_confidence is not None:
        coordinate_confidence = np.asarray(coordinate_confidence, float)
        if len(coordinate_confidence) == len(cones):
            # compute minimum distance from each cone to any path segment
            n_segments = len(path) - 1
            min_cone_dist = np.full(len(cones), np.inf)
            for idx in range(n_segments):
                A = path[idx]
                B = path[idx + 1]
                vec_AB = B - A
                seg_len_sq = np.dot(vec_AB, vec_AB)
                if seg_len_sq < 1e-10:
                    continue
                vec_AC = cones - A
                t = np.clip(np.dot(vec_AC, vec_AB) / seg_len_sq, 0, 1)
                proj = A + t[:, np.newaxis] * vec_AB
                dist = np.linalg.norm(cones - proj, axis=1)
                min_cone_dist = np.minimum(min_cone_dist, dist)

            # only consider cones within 2x track width of the path
            near_mask = min_cone_dist < 6.0
            if np.any(near_mask):
                # mean probability radius of nearby cones
                coord_conf_metric = float(np.mean(coordinate_confidence[near_mask]) ** 2)

# normalization constants (expected values from AMZ paper)
    EXPECTED_MAX_ANGLE = np.pi / 4  # ~45 degrees max turn
    EXPECTED_SPACING_VAR = 1.0      # ~1m variance in cone spacing
    EXPECTED_TRACKWIDTH_VAR = 0.5   # ~0.5m variance in track width (should be consistent ~3m)
    EXPECTED_COLOR_UNCERTAINTY = 0.25  # 25% average uncertainty
    EXPECTED_LENGTH_DEV = 2.0       # ~2m deviation from target
    EXPECTED_COORD_RADIUS = 0.5     # ~0.5m average positional uncertainty

# normalized metrics
    norm_angle = angle_metric / EXPECTED_MAX_ANGLE if EXPECTED_MAX_ANGLE > 0 else angle_metric
    norm_smoothness = smoothness_metric / EXPECTED_MAX_ANGLE if EXPECTED_MAX_ANGLE > 0 else smoothness_metric
    norm_spacing = spacing_metric / EXPECTED_SPACING_VAR if EXPECTED_SPACING_VAR > 0 else spacing_metric
    norm_trackwidth = trackwidth_metric / EXPECTED_TRACKWIDTH_VAR if EXPECTED_TRACKWIDTH_VAR > 0 else trackwidth_metric
    norm_color = color_metric / EXPECTED_COLOR_UNCERTAINTY if EXPECTED_COLOR_UNCERTAINTY > 0 else color_metric
    norm_length = length_metric / EXPECTED_LENGTH_DEV if EXPECTED_LENGTH_DEV > 0 else length_metric
    norm_coord = coord_conf_metric / EXPECTED_COORD_RADIUS if EXPECTED_COORD_RADIUS > 0 else coord_conf_metric

# total cost
    cost = (
        2.5 * norm_angle           # qc: penalize sharp turns
        + 1.5 * norm_smoothness    # curve smoothness
        + 1.0 * norm_spacing       # qs: cone spacing consistency
        + 5 * norm_trackwidth    # qw: track width consistency (AMZ metric!)
        + 2 * norm_color         # qcolor: color confidence
        + 1 * norm_length        # ql: path length deviation
        + 10 * boundary_violation_metric  # high penalty for wrong side (not normalized, already 0-1)
        + 2.0 * norm_coord      # qcoord: positional uncertainty
    )
    return float(cost)
