"""Unit tests for cost_functions.py"""
import numpy as np
import pytest
from cost_functions import (
    evaluate_path_cost,
    assign_cones_to_segments,
    count_boundary_violations,
    calculate_boundary_violation,
    calculate_trackwidth_variance,
    BIG_COST
)
import config as cfg


def create_simple_path():
    """Create a simple straight path."""
    return np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 0.0],
        [15.0, 0.0],
    ])


def create_simple_cones():
    """Create cones for a simple straight track."""
    cones = np.array([
        [2.5, 1.5],   # blue (left)
        [2.5, -1.5],  # yellow (right)
        [7.5, 1.5],   # blue (left)
        [7.5, -1.5],  # yellow (right)
        [12.5, 1.5],  # blue (left)
        [12.5, -1.5], # yellow (right)
    ])
    colors = np.array([
        [0.95, 0.02, 0.02, 0.01],  # blue
        [0.02, 0.95, 0.02, 0.01],  # yellow
        [0.95, 0.02, 0.02, 0.01],  # blue
        [0.02, 0.95, 0.02, 0.01],  # yellow
        [0.95, 0.02, 0.02, 0.01],  # blue
        [0.02, 0.95, 0.02, 0.01],  # yellow
    ])
    return cones, colors


class TestEvaluatePathCost:
    """Tests for evaluate_path_cost function."""

    def test_returns_float(self):
        """Cost should be a float."""
        path = create_simple_path()
        cones, colors = create_simple_cones()
        coordinate_confidence = np.zeros(len(cones))

        cost = evaluate_path_cost(path, cones, coordinate_confidence, colors)

        assert isinstance(cost, float)

    def test_cost_is_non_negative(self):
        """Cost should be non-negative."""
        path = create_simple_path()
        cones, colors = create_simple_cones()
        coordinate_confidence = np.zeros(len(cones))

        cost = evaluate_path_cost(path, cones, coordinate_confidence, colors)

        assert cost >= 0

    def test_short_path_returns_big_cost(self):
        """Path with fewer than 2 points should return BIG_COST."""
        path = np.array([[0.0, 0.0]])
        cones, colors = create_simple_cones()
        coordinate_confidence = np.zeros(len(cones))

        cost = evaluate_path_cost(path, cones, coordinate_confidence, colors)

        assert cost == BIG_COST

    def test_too_few_cones_returns_big_cost(self):
        """Too few cones should return BIG_COST."""
        path = create_simple_path()
        cones = np.array([[0.0, 1.5], [0.0, -1.5]])  # only 2 cones
        colors = np.array([
            [0.95, 0.02, 0.02, 0.01],
            [0.02, 0.95, 0.02, 0.01],
        ])
        coordinate_confidence = np.zeros(len(cones))

        cost = evaluate_path_cost(path, cones, coordinate_confidence, colors)

        assert cost == BIG_COST

    def test_works_without_colors(self):
        """Should work when colors is None."""
        path = create_simple_path()
        cones, _ = create_simple_cones()
        coordinate_confidence = np.zeros(len(cones))

        cost = evaluate_path_cost(path, cones, coordinate_confidence, colors=None)

        assert isinstance(cost, float)
        assert cost >= 0
        assert cost < BIG_COST

    def test_straight_path_lower_cost_than_zigzag(self):
        """Straight path should have lower cost than zigzag path."""
        cones, colors = create_simple_cones()

        straight_path = create_simple_path()
        zigzag_path = np.array([
            [0.0, 0.0],
            [5.0, 1.0],
            [10.0, -1.0],
            [15.0, 0.5],
        ])
        coordinate_confidence = np.zeros(len(cones))

        straight_cost = evaluate_path_cost(straight_path, cones, coordinate_confidence, colors)
        zigzag_cost = evaluate_path_cost(zigzag_path, cones, coordinate_confidence, colors)

        assert straight_cost < zigzag_cost


class TestAssignConesToSegments:
    """Tests for assign_cones_to_segments function."""

    def test_returns_two_arrays(self):
        """Should return segment assignments and side assignments."""
        path = create_simple_path()
        cones, _ = create_simple_cones()

        seg_assign, side_assign = assign_cones_to_segments(path, cones)

        assert isinstance(seg_assign, np.ndarray)
        assert isinstance(side_assign, np.ndarray)

    def test_assignment_length_matches_cones(self):
        """Assignment arrays should have same length as cones."""
        path = create_simple_path()
        cones, _ = create_simple_cones()

        seg_assign, side_assign = assign_cones_to_segments(path, cones)

        assert len(seg_assign) == len(cones)
        assert len(side_assign) == len(cones)

    def test_side_assignments_are_valid(self):
        """Side assignments should be -1, 0, or 1."""
        path = create_simple_path()
        cones, _ = create_simple_cones()

        seg_assign, side_assign = assign_cones_to_segments(path, cones)

        for side in side_assign:
            assert side in [-1, 0, 1]

    def test_left_cones_assigned_left(self):
        """Cones on the left (positive y) should have side=1."""
        path = create_simple_path()
        cones, _ = create_simple_cones()

        seg_assign, side_assign = assign_cones_to_segments(path, cones)

        # Cones at indices 0, 2, 4 are on the left (y=1.5)
        assert side_assign[0] == 1
        assert side_assign[2] == 1
        assert side_assign[4] == 1

    def test_right_cones_assigned_right(self):
        """Cones on the right (negative y) should have side=-1."""
        path = create_simple_path()
        cones, _ = create_simple_cones()

        seg_assign, side_assign = assign_cones_to_segments(path, cones)

        # Cones at indices 1, 3, 5 are on the right (y=-1.5)
        assert side_assign[1] == -1
        assert side_assign[3] == -1
        assert side_assign[5] == -1


class TestCountBoundaryViolations:
    """Tests for count_boundary_violations function."""

    def test_no_violations_correct_layout(self):
        """Should return 0 when blue on left, yellow on right."""
        # Blue = left = side 1, Yellow = right = side -1
        cone_colors = np.array([
            cfg.CONE_COLOR_BLUE,
            cfg.CONE_COLOR_YELLOW,
            cfg.CONE_COLOR_BLUE,
            cfg.CONE_COLOR_YELLOW,
        ])
        side_assignments = np.array([1, -1, 1, -1])

        violations = count_boundary_violations(cone_colors, side_assignments)

        assert violations == 0

    def test_violations_wrong_layout(self):
        """Should count violations when colors on wrong side."""
        # Blue on right (should be left), Yellow on left (should be right)
        cone_colors = np.array([
            cfg.CONE_COLOR_BLUE,
            cfg.CONE_COLOR_YELLOW,
        ])
        side_assignments = np.array([-1, 1])  # wrong sides

        violations = count_boundary_violations(cone_colors, side_assignments)

        assert violations == 2

    def test_ignores_orange_cones(self):
        """Should ignore orange cones."""
        cone_colors = np.array([
            cfg.CONE_COLOR_ORANGE_SMALL,
            cfg.CONE_COLOR_ORANGE_LARGE,
        ])
        side_assignments = np.array([1, -1])

        violations = count_boundary_violations(cone_colors, side_assignments)

        assert violations == 0


class TestCalculateTrackwidthVariance:
    """Tests for calculate_trackwidth_variance function."""

    def test_returns_float(self):
        """Should return a float."""
        path = create_simple_path()
        cones, colors = create_simple_cones()

        variance = calculate_trackwidth_variance(path, cones, colors)

        assert isinstance(variance, float)

    def test_consistent_width_low_variance(self):
        """Consistent track width should give low variance."""
        path = create_simple_path()
        cones, colors = create_simple_cones()

        variance = calculate_trackwidth_variance(path, cones, colors)

        # With consistent 3m width, variance should be low
        assert variance < 1.0

    def test_handles_empty_colors(self):
        """Should return 0 for empty colors."""
        path = create_simple_path()
        cones, _ = create_simple_cones()

        variance = calculate_trackwidth_variance(path, cones, colors=None)

        assert variance == 0.0
