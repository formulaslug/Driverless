import numpy as np
import config as cfg

# Most likely color per cone, or CONE_COLOR_UNKNOWN when no color is confident enough.
def label_cones(colors):
    colors = np.asarray(colors, float)
    labels = np.argmax(colors, axis=1)
    labels[np.max(colors, axis=1) < cfg.MIN_COLOR_CONFIDENCE] = cfg.CONE_COLOR_UNKNOWN
    return labels

# Track boundary per cone: CONE_COLOR_BLUE (left), CONE_COLOR_YELLOW (right) or
# CONE_COLOR_UNKNOWN. Orange cones are given the boundary of the nearest
# blue/yellow cone within ORANGE_NEIGHBOR_RADIUS; unknown cones stay unknown.
def boundary_sides(cones, colors):
    labels = label_cones(colors)
    sides = np.where(np.isin(labels, [cfg.CONE_COLOR_BLUE, cfg.CONE_COLOR_YELLOW]), labels, cfg.CONE_COLOR_UNKNOWN)
    isOrange = np.isin(labels, [cfg.CONE_COLOR_ORANGE_SMALL, cfg.CONE_COLOR_ORANGE_LARGE])
    isKnown = sides != cfg.CONE_COLOR_UNKNOWN

    if np.any(isOrange) and np.any(isKnown):
        cones = np.asarray(cones, float)
        distances = np.linalg.norm(cones[isOrange][:, None, :] - cones[isKnown][None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        withinRadius = distances[np.arange(len(nearest)), nearest] <= cfg.ORANGE_NEIGHBOR_RADIUS
        sides[np.flatnonzero(isOrange)[withinRadius]] = sides[isKnown][nearest[withinRadius]]

    return sides
