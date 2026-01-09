
# heuristic cost function for evaluating best candidate path based on 5 metrics (AMZ)
#    1. angle sharpness   - penalizes big/sudden turns unless necessary 
#    2. smooth curves - penalizes randomized direction changes
#    3. cone spacing - cones on each boundary are usually evenly spaced
#    4. color confidence - penalizes cones whose color classification is uncertain or wrong 
#    5. path length - prefers paths witha 12m range can be changeable 
# for refrence 1 and 2 are combined in angle + curve metric

import numpy as np
import config as cfg

TARGET_PATH_LENGTH = 12.0   # desired horizon
BIG_COST = 1e9

def evaluate_path_cost(path, cones, colors=None):
    path = np.asarray(path, float)
    cones = np.asarray(cones, float)

    if len(path) < 2 or len(cones) < cfg.MIN_CONES_FOR_VALID_PATH:
        return BIG_COST

#angle + curve metric
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

#cone spacing metric
    if len(cones) > 1:
        nn_dist = []
        for i in range(len(cones)):
            d = np.linalg.norm(cones[i] - cones, axis=1)
            d[i] = np.inf
            nn_dist.append(np.min(d))
        spacing_metric = float(np.var(nn_dist))
    else:
        spacing_metric = 0.0

#color metric
    color_metric = 0.0
    if colors is not None:
        colors = np.asarray(colors, float)
        if colors.shape[0] == cones.shape[0]:
            p_best = np.max(colors, axis=1)   # most likely color
            wrong_p = 1.0 - p_best           # chance of being wrong
            color_metric = float(np.max(wrong_p)**2)

#lenght metric
    seg_lengths = np.linalg.norm(segs, axis=1)
    total_length = float(np.sum(seg_lengths))
    length_metric = (total_length - TARGET_PATH_LENGTH) ** 2

#total cost 
    cost = (
        2.0 * angle_metric
        + 1.0 * smoothness_metric
        + 1.0 * spacing_metric
        + 2.0 * color_metric
        + 0.5 * length_metric
    )
    return float(cost)
