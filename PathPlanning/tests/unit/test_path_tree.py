"""Unit tests for path_tree.py"""
import numpy as np
import pytest
from path_tree import is_forward, find_nearest_waypoints, get_path_tree


def create_simple_track():
    """Create a simple straight track with blue cones on left, yellow on right."""
    cones = np.array([
        [0.0, 1.5],   # blue
        [0.0, -1.5],  # yellow
        [5.0, 1.5],   # blue
        [5.0, -1.5],  # yellow
        [10.0, 1.5],  # blue
        [10.0, -1.5], # yellow
        [15.0, 1.5],  # blue
        [15.0, -1.5], # yellow
    ])
    colors = np.array([
        [0.95, 0.02, 0.02, 0.01],
        [0.02, 0.95, 0.02, 0.01],
        [0.95, 0.02, 0.02, 0.01],
        [0.02, 0.95, 0.02, 0.01],
        [0.95, 0.02, 0.02, 0.01],
        [0.02, 0.95, 0.02, 0.01],
        [0.95, 0.02, 0.02, 0.01],
        [0.02, 0.95, 0.02, 0.01],
    ])
    return cones, colors


class TestIsForward:
    """Tests for is_forward function."""

    def test_waypoint_directly_ahead(self):
        """Waypoint directly in front should be forward."""
        vehicle_pos = [0.0, 0.0]
        vehicle_heading = 0.0  # facing +x direction
        waypoint = [5.0, 0.0]

        assert is_forward(waypoint, vehicle_pos, vehicle_heading) == True

    def test_waypoint_directly_behind(self):
        """Waypoint directly behind should not be forward."""
        vehicle_pos = [0.0, 0.0]
        vehicle_heading = 0.0  # facing +x direction
        waypoint = [-5.0, 0.0]

        assert is_forward(waypoint, vehicle_pos, vehicle_heading) == False

    def test_waypoint_to_left_forward(self):
        """Waypoint ahead and to the left (within 90 deg) should be forward."""
        vehicle_pos = [0.0, 0.0]
        vehicle_heading = 0.0  # facing +x direction
        waypoint = [5.0, 3.0]  # ahead and left

        assert is_forward(waypoint, vehicle_pos, vehicle_heading) == True

    def test_waypoint_to_right_forward(self):
        """Waypoint ahead and to the right (within 90 deg) should be forward."""
        vehicle_pos = [0.0, 0.0]
        vehicle_heading = 0.0  # facing +x direction
        waypoint = [5.0, -3.0]  # ahead and right

        assert is_forward(waypoint, vehicle_pos, vehicle_heading) == True

    def test_waypoint_perpendicular_left(self):
        """Waypoint exactly perpendicular (90 deg) should not be forward."""
        vehicle_pos = [0.0, 0.0]
        vehicle_heading = 0.0  # facing +x direction
        waypoint = [0.0, 5.0]  # directly to the left

        # dot product is 0, so is_forward returns False (> 0 required)
        assert is_forward(waypoint, vehicle_pos, vehicle_heading) == False

    def test_different_heading(self):
        """Test with vehicle facing +y direction."""
        vehicle_pos = [0.0, 0.0]
        vehicle_heading = np.pi / 2  # facing +y direction

        assert is_forward([0.0, 5.0], vehicle_pos, vehicle_heading) == True
        assert is_forward([0.0, -5.0], vehicle_pos, vehicle_heading) == False


class TestFindNearestWaypoints:
    """Tests for find_nearest_waypoints function."""

    def test_returns_list(self):
        """Should return a list of waypoints."""
        waypoints = np.array([[5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
        vehicle_pos = [0.0, 0.0]
        vehicle_heading = 0.0

        result = find_nearest_waypoints(vehicle_pos, vehicle_heading, waypoints, k=2)

        assert isinstance(result, list)

    def test_returns_k_waypoints(self):
        """Should return exactly k waypoints if available."""
        waypoints = np.array([[5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
        vehicle_pos = [0.0, 0.0]
        vehicle_heading = 0.0

        result = find_nearest_waypoints(vehicle_pos, vehicle_heading, waypoints, k=2)

        assert len(result) == 2

    def test_returns_nearest_waypoints(self):
        """Should return the k nearest waypoints."""
        waypoints = np.array([[5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
        vehicle_pos = [0.0, 0.0]
        vehicle_heading = 0.0

        result = find_nearest_waypoints(vehicle_pos, vehicle_heading, waypoints, k=1)

        # Nearest should be [5.0, 0.0]
        assert result[0] == (5.0, 0.0)

    def test_filters_backward_waypoints(self):
        """Should not return waypoints behind the vehicle."""
        waypoints = np.array([[-5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
        vehicle_pos = [0.0, 0.0]
        vehicle_heading = 0.0

        result = find_nearest_waypoints(vehicle_pos, vehicle_heading, waypoints, k=3)

        # Should not include [-5.0, 0.0]
        assert (-5.0, 0.0) not in result


class TestGetPathTree:
    """Tests for get_path_tree function."""

    def test_returns_list_of_paths(self):
        """Should return a list of paths."""
        cones, colors = create_simple_track()
        coordinate_confidence = np.zeros(len(cones))
        vehicle_pos = [-2.0, 0.0]
        vehicle_heading = 0.0

        paths = get_path_tree(cones, coordinate_confidence, colors, vehicle_pos, vehicle_heading,
                              max_depth=3, k_start=2)

        assert isinstance(paths, list)

    def test_paths_have_waypoints(self):
        """Paths should contain waypoints."""
        cones, colors = create_simple_track()
        coordinate_confidence = np.zeros(len(cones))
        vehicle_pos = [-2.0, 0.0]
        vehicle_heading = 0.0

        paths = get_path_tree(cones, coordinate_confidence, colors, vehicle_pos, vehicle_heading,
                              max_depth=5, k_start=3)

        if len(paths) > 0:
            # Each path should have at least 1 waypoint
            for path in paths:
                assert len(path) >= 1

    def test_no_duplicate_waypoints_in_path(self):
        """Each path should not visit the same waypoint twice."""
        cones, colors = create_simple_track()
        coordinate_confidence = np.zeros(len(cones))
        vehicle_pos = [-2.0, 0.0]
        vehicle_heading = 0.0

        paths = get_path_tree(cones, coordinate_confidence, colors, vehicle_pos, vehicle_heading,
                              max_depth=5, k_start=3)

        for path in paths:
            # Check no duplicates
            assert len(path) == len(set(path)), "Path contains duplicate waypoints"
