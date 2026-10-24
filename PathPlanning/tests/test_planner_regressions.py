import numpy as np
import config as cfg
from cone_labels import label_cones, boundary_sides
from path_planner import plan_path
from path_tree import get_path_tree
from cost_functions import assign_cones_to_segments

BLUE = [0.9, 0.05, 0.03, 0.02]
YELLOW = [0.05, 0.9, 0.03, 0.02]
UNKNOWN = [0.25, 0.25, 0.25, 0.25]
ORANGE = [0.03, 0.03, 0.04, 0.9]

def straightTrack(xs, width=3.0):
    cones, colors = [], []
    for x in xs:
        cones += [[x, width / 2], [x, -width / 2]]
        colors += [BLUE, YELLOW]
    return np.array(cones, float), np.array(colors, float)

vehiclePos = np.array([-2.0, 0.0])
xs = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

def testUniformProbabilitiesAreUnknownNotBlue():
    assert label_cones(np.array([UNKNOWN]))[0] == cfg.CONE_COLOR_UNKNOWN
    assert label_cones(np.array([BLUE]))[0] == cfg.CONE_COLOR_BLUE

def testUnknownConesNeverFormBoundaries():
    cones = np.array([[0, 1.5], [0, -1.5], [5, 1.5]], float)
    colors = np.array([BLUE, YELLOW, UNKNOWN], float)
    sides = boundary_sides(cones, colors)
    assert sides[2] == cfg.CONE_COLOR_UNKNOWN

def testOrangeTakesSideOfNearestBoundaryCone():
    cones = np.array([[0, 1.5], [0, -1.5], [3, 1.5], [3, -1.5]], float)
    colors = np.array([BLUE, YELLOW, ORANGE, ORANGE], float)
    sides = boundary_sides(cones, colors)
    assert sides[2] == cfg.CONE_COLOR_BLUE
    assert sides[3] == cfg.CONE_COLOR_YELLOW

def testPlanSurvivesUnknownCones():
    cones, colors = straightTrack(xs)
    colors[[3, 8]] = UNKNOWN
    smoothPath, curvature = plan_path(cones, np.zeros(len(cones)), colors, vehiclePos, 0.0)
    assert smoothPath is not None and curvature is not None

def testPlanThroughOrangeGate():
    cones, colors = straightTrack(xs)
    gate = [i for i, c in enumerate(cones) if c[0] == 10]
    colors[gate] = ORANGE
    smoothPath, _ = plan_path(cones, np.zeros(len(cones)), colors, vehiclePos, 0.0)
    assert smoothPath is not None
    assert smoothPath[:, 0].max() > 15, "path should continue past the orange gate"
    assert np.all(np.abs(smoothPath[:, 1]) < 1.5)

def testPathStartsAtVehicle():
    cones, colors = straightTrack(xs)
    smoothPath, _ = plan_path(cones, np.zeros(len(cones)), colors, vehiclePos, 0.0)
    assert np.allclose(smoothPath[0], vehiclePos, atol=1e-6)

def testPathTreeHasNoDuplicates():
    cones, colors = straightTrack(xs)
    paths = get_path_tree(cones, np.zeros(len(cones)), colors, vehiclePos, 0.0, cfg.MAX_TREE_DEPTH, cfg.K_START)
    assert len(paths) > 0
    assert len({tuple(p) for p in paths}) == len(paths)

def testCollinearConesReturnNone():
    cones = np.array([[x, 0.0] for x in range(6)])
    colors = np.array([BLUE, YELLOW] * 3, float)
    assert plan_path(cones, np.zeros(6), colors, np.array([-2.0, 0.0]), 0.0) == (None, None)

def testFarConesAreNotAssignedToPath():
    path = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    cones = np.array([[5.0, 1.5], [5.0, 40.0]])
    segments, sides = assign_cones_to_segments(path, cones)
    assert segments[0] >= 0 and sides[0] == 1
    assert segments[1] == -1 and sides[1] == 0

def testAllUnknownConesReturnNone():
    cones, colors = straightTrack(xs)
    colors[:] = UNKNOWN
    assert plan_path(cones, np.zeros(len(cones)), colors, vehiclePos, 0.0) == (None, None)

def testVehicleOnFirstWaypointStillSmooths():
    cones, colors = straightTrack(xs)
    smoothPath, _ = plan_path(cones, np.zeros(len(cones)), colors, np.array([0.0, 0.0]), 0.0)
    assert smoothPath is not None
