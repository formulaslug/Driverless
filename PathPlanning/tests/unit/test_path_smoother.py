"""Unit tests for path_smoother.py"""
import numpy as np
import pytest
from path_smoother import smooth_path


def create_straight_waypoints():
    """Create waypoints for a straight path."""
    return np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 0.0],
        [15.0, 0.0],
    ])


def create_curved_waypoints():
    """Create waypoints for a curved path (90 degree turn)."""
    return np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 2.0],
        [12.0, 7.0],
        [12.0, 12.0],
    ])


class TestSmoothPath:
    """Tests for smooth_path function."""

    def test_returns_tuple(self):
        """Should return a tuple of (path, curvature)."""
        waypoints = create_straight_waypoints()
        result = smooth_path(waypoints)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_path_is_numpy_array(self):
        """Smoothed path should be a numpy array."""
        waypoints = create_straight_waypoints()
        path, curvature = smooth_path(waypoints)

        assert isinstance(path, np.ndarray)

    def test_curvature_is_numpy_array(self):
        """Curvature should be a numpy array."""
        waypoints = create_straight_waypoints()
        path, curvature = smooth_path(waypoints)

        assert isinstance(curvature, np.ndarray)

    def test_path_shape(self):
        """Path should be (n_points, 2) for x,y coordinates."""
        waypoints = create_straight_waypoints()
        path, curvature = smooth_path(waypoints)

        assert len(path.shape) == 2
        assert path.shape[1] == 2

    def test_curvature_shape(self):
        """Curvature should be 1D array."""
        waypoints = create_straight_waypoints()
        path, curvature = smooth_path(waypoints)

        assert len(curvature.shape) == 1

    def test_path_starts_near_first_waypoint(self):
        """Smoothed path should start near the first waypoint."""
        waypoints = create_straight_waypoints()
        path, curvature = smooth_path(waypoints)

        start_distance = np.linalg.norm(path[0] - waypoints[0])
        assert start_distance < 1.0  # within 1 meter

    def test_path_ends_near_last_waypoint(self):
        """Smoothed path should end near the last waypoint."""
        waypoints = create_straight_waypoints()
        path, curvature = smooth_path(waypoints)

        end_distance = np.linalg.norm(path[-1] - waypoints[-1])
        assert end_distance < 1.0  # within 1 meter


class TestCurvature:
    """Tests for curvature calculation."""

    def test_straight_path_low_curvature(self):
        """Straight path should have near-zero curvature."""
        waypoints = create_straight_waypoints()
        path, curvature = smooth_path(waypoints)

        # Curvature should be very small for straight line
        max_curvature = np.max(np.abs(curvature))
        assert max_curvature < 0.1

    def test_curved_path_has_curvature(self):
        """Curved path should have non-zero curvature."""
        waypoints = create_curved_waypoints()
        path, curvature = smooth_path(waypoints)

        # Should have some curvature
        max_curvature = np.max(np.abs(curvature))
        assert max_curvature > 0.01

    def test_curvature_values_reasonable(self):
        """Curvature values should be within reasonable range."""
        waypoints = create_curved_waypoints()
        path, curvature = smooth_path(waypoints)

        # For a 9m min turn diameter, max curvature ~0.22 (1/4.5m radius)
        # Allow some margin for the test
        max_curvature = np.max(np.abs(curvature))
        assert max_curvature < 1.0  # curvature shouldn't exceed 1/1m radius
