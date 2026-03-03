"""Unit tests for delaunay.py"""
import numpy as np
import pytest
from delaunay import get_midpoints


def create_simple_track():
    """Create a simple straight track with blue cones on left, yellow on right."""
    # Blue cones on left (y = 1.5)
    # Yellow cones on right (y = -1.5)
    cones = np.array([
        [0.0, 1.5],   # blue
        [0.0, -1.5],  # yellow
        [5.0, 1.5],   # blue
        [5.0, -1.5],  # yellow
        [10.0, 1.5],  # blue
        [10.0, -1.5], # yellow
    ])
    # Color probabilities: [blue, yellow, orange_small, orange_large]
    colors = np.array([
        [0.95, 0.02, 0.02, 0.01],  # blue
        [0.02, 0.95, 0.02, 0.01],  # yellow
        [0.95, 0.02, 0.02, 0.01],  # blue
        [0.02, 0.95, 0.02, 0.01],  # yellow
        [0.95, 0.02, 0.02, 0.01],  # blue
        [0.02, 0.95, 0.02, 0.01],  # yellow
    ])
    return cones, colors


class TestGetMidpoints:
    """Tests for get_midpoints function."""

    def test_returns_tuple_of_three(self):
        """get_midpoints should return (waypoints, graph, triangulation)."""
        cones, colors = create_simple_track()
        result = get_midpoints(cones, colors)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_waypoints_are_numpy_array(self):
        """Waypoints should be a numpy array."""
        cones, colors = create_simple_track()
        waypoints, graph, tri = get_midpoints(cones, colors)

        assert isinstance(waypoints, np.ndarray)

    def test_waypoints_have_correct_shape(self):
        """Waypoints should be (n, 2) array of x,y coordinates."""
        cones, colors = create_simple_track()
        waypoints, graph, tri = get_midpoints(cones, colors)

        assert len(waypoints.shape) == 2
        assert waypoints.shape[1] == 2

    def test_graph_is_dict(self):
        """Graph should be a dictionary."""
        cones, colors = create_simple_track()
        waypoints, graph, tri = get_midpoints(cones, colors)

        assert isinstance(graph, dict)

    def test_midpoints_are_actually_midpoints(self):
        """Each waypoint should be roughly between two cones."""
        cones, colors = create_simple_track()
        waypoints, graph, tri = get_midpoints(cones, colors)

        # For a 3m wide track, waypoints should be near the center (y ~ 0)
        for wp in waypoints:
            # Waypoint y-coordinate should be between -1.5 and 1.5
            assert -2.0 <= wp[1] <= 2.0


class TestGraphConnectivity:
    """Tests for waypoint graph connectivity."""

    def test_graph_values_are_sets(self):
        """Graph values should be sets of connected waypoints."""
        cones, colors = create_simple_track()
        waypoints, graph, tri = get_midpoints(cones, colors)

        for key, value in graph.items():
            assert isinstance(value, set)

    def test_graph_is_bidirectional(self):
        """If A connects to B, B should connect to A."""
        cones, colors = create_simple_track()
        waypoints, graph, tri = get_midpoints(cones, colors)

        for wp1, connections in graph.items():
            for wp2 in connections:
                assert wp1 in graph.get(wp2, set()), \
                    f"Graph not bidirectional: {wp1} -> {wp2} but not reverse"
