
# heuristic cost function for evaluating best candidate path based on 5 metrics (AMZ)
#    1. angle sharpness   - penalizes big/sudden turns unless necessary 
#    2. smooth curves - penalizes randomized direction changes
#    3. cone spacing - cones on each boundary are usually evenly spaced
#    4. color confidence - penalizes cones whose color classification is uncertain or wrong 
#    5. path length - prefers paths witha 12m range can be changeable 
# for refrence 1 and 2 are combined in angle + curve metric

import numpy as np
import config as cfg
from cone_labels import boundary_sides
from scipy.spatial.distance import pdist, squareform

TARGET_PATH_LENGTH = 15.0   # desired horizon
BIG_COST = 1e9

def assign_cones_to_segments(path, cones):
    # returns (segment_assignments, side_assignments); cones farther than
    # MAX_CONE_TO_PATH_DISTANCE from every segment get segment -1 and side 0
    path = np.asarray(path, float)
    cones = np.asarray(cones, float)

    starts = path[:-1]
    vec_AB = path[1:] - starts                              # (n_segments, 2)
    seg_len_sq = np.sum(vec_AB**2, axis=1)
    degenerate = seg_len_sq < 1e-10

    vec_AC = cones[:, np.newaxis, :] - starts[np.newaxis]   # (n_cones, n_segments, 2)
    t = np.sum(vec_AC * vec_AB[np.newaxis], axis=2) / np.where(degenerate, 1.0, seg_len_sq)
    t = np.clip(t, 0.0, 1.0)

    projections = starts[np.newaxis] + t[:, :, np.newaxis] * vec_AB[np.newaxis]
    distances = np.linalg.norm(cones[:, np.newaxis, :] - projections, axis=2)
    distances[:, degenerate] = np.inf

    # positive cross product: cone is on the left of the segment
    cross_products = vec_AB[np.newaxis, :, 0] * vec_AC[:, :, 1] - vec_AB[np.newaxis, :, 1] * vec_AC[:, :, 0]

    rows = np.arange(len(cones))
    segment_assignments = np.argmin(distances, axis=1)
    unassigned = distances[rows, segment_assignments] > cfg.MAX_CONE_TO_PATH_DISTANCE

    cross = cross_products[rows, segment_assignments]
    side_assignments = np.where(cross > 1e-6, 1, np.where(cross < -1e-6, -1, 0))
    side_assignments[unassigned] = 0
    segment_assignments[unassigned] = -1

    return segment_assignments, side_assignments

def count_boundary_violations(cone_colors, side_assignments):
    # only blue/yellow boundary cones with a left or right side assignment count
    valid_mask = np.isin(cone_colors, [cfg.CONE_COLOR_BLUE, cfg.CONE_COLOR_YELLOW]) & \
                    (side_assignments != 0)
                    
    expected_sides = np.zeros_like(cone_colors, dtype=int)
    expected_sides[cone_colors == cfg.CONE_COLOR_BLUE] = 1 # left side cones should be blue
    expected_sides[cone_colors == cfg.CONE_COLOR_YELLOW] = -1 # right side cones should be yellow
    
    mismatches = (expected_sides != side_assignments) & valid_mask
    return int(np.sum(mismatches))

def calculate_boundary_violation(path, cones, colors, assignments=None):
    boundaries = boundary_sides(cones, colors)
    if assignments is None:
        assignments = assign_cones_to_segments(path, cones)
    segment_assignments, side_assignments = assignments
    violations = count_boundary_violations(boundaries, side_assignments)

    n_scored = np.sum(np.isin(boundaries, [cfg.CONE_COLOR_BLUE, cfg.CONE_COLOR_YELLOW]) & (side_assignments != 0))

    if n_scored == 0:
        return 0.0

    return float(violations) / float(n_scored)

def calculate_trackwidth_variance(path, cones, colors, assignments=None):
    if colors is None or len(colors) == 0:
        return 0.0

    cone_colors = boundary_sides(cones, colors)
    if assignments is None:
        assignments = assign_cones_to_segments(path, cones)
    segment_assignments, side_assignments = assignments

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

    # cones close enough to this path to be scored against it
    segment_assignments, side_assignments = assign_cones_to_segments(path, cones)
    near_mask = segment_assignments >= 0
    near_cones = cones[near_mask]

    # cone spacing metric: evenness of nearest-neighbor distances along the path
    if len(near_cones) > 2:
        dist_matrix = squareform(pdist(near_cones))
        np.fill_diagonal(dist_matrix, np.inf)
        spacing_metric = float(np.var(np.min(dist_matrix, axis=1)))
    else:
        spacing_metric = 0.0

    # color metric
    color_metric = 0.0
    if colors is not None:
        colors = np.asarray(colors, float)
        if colors.shape[0] == cones.shape[0] and np.any(near_mask):
            p_best = np.max(colors[near_mask], axis=1)   # most likely color
            wrong_p = 1.0 - p_best                       # chance of being wrong
            color_metric = float(np.mean(wrong_p)**2)

    # length metric
    seg_lengths = np.linalg.norm(segs, axis=1)
    total_length = float(np.sum(seg_lengths))
    length_metric = (total_length - TARGET_PATH_LENGTH) ** 2

    # track width variance and track boundary metrics
    trackwidth_metric = 0.0
    boundary_violation_metric = 0.0
    if colors is not None:
        assignments = (segment_assignments, side_assignments)
        trackwidth_metric = calculate_trackwidth_variance(path, cones, colors, assignments)
        boundary_violation_metric = calculate_boundary_violation(path, cones, colors, assignments)

    # coordinate confidence metric: mean positional uncertainty radius of scored cones
    coord_conf_metric = 0.0
    if coordinate_confidence is not None:
        coordinate_confidence = np.asarray(coordinate_confidence, float)
        if len(coordinate_confidence) == len(cones) and np.any(near_mask):
            coord_conf_metric = float(np.mean(coordinate_confidence[near_mask]) ** 2)

    # normalization constants (expected values from AMZ paper); every metric is
    # a squared quantity, so it is divided by the square of its expected scale
    EXPECTED_MAX_ANGLE = np.pi / 4  # ~45 degrees max turn
    EXPECTED_SPACING_VAR = 1.0      # ~1m^2 variance in cone spacing
    EXPECTED_TRACKWIDTH_VAR = 0.5   # ~0.5m^2 variance in track width (should be consistent ~3m)
    EXPECTED_COLOR_UNCERTAINTY = 0.25  # 25% average uncertainty
    EXPECTED_LENGTH_DEV = 2.0       # ~2m deviation from target
    EXPECTED_COORD_RADIUS = 0.5     # ~0.5m average positional uncertainty

    norm_angle = angle_metric / EXPECTED_MAX_ANGLE**2
    norm_smoothness = smoothness_metric / EXPECTED_MAX_ANGLE**2
    norm_spacing = spacing_metric / EXPECTED_SPACING_VAR
    norm_trackwidth = trackwidth_metric / EXPECTED_TRACKWIDTH_VAR
    norm_color = color_metric / EXPECTED_COLOR_UNCERTAINTY**2
    norm_length = length_metric / EXPECTED_LENGTH_DEV**2
    norm_coord = coord_conf_metric / EXPECTED_COORD_RADIUS**2

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
