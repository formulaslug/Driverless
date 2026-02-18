"""Unit tests for beam_search.py"""
import numpy as np
import pytest
from beam_search import beam_search_prune


def create_simple_cones():
    """Create cones for a simple straight track."""
    cones = np.array([
        [2.5, 1.5],
        [2.5, -1.5],
        [7.5, 1.5],
        [7.5, -1.5],
        [12.5, 1.5],
        [12.5, -1.5],
    ])
    colors = np.array([
        [0.95, 0.02, 0.02, 0.01],
        [0.02, 0.95, 0.02, 0.01],
        [0.95, 0.02, 0.02, 0.01],
        [0.02, 0.95, 0.02, 0.01],
        [0.95, 0.02, 0.02, 0.01],
        [0.02, 0.95, 0.02, 0.01],
    ])
    return cones, colors


class TestBeamSearchPrune:
    """Tests for beam_search_prune function."""

    def test_returns_list(self):
        """Should return a list of paths."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0)],
            [(0.0, 0.0), (5.0, 0.5), (10.0, 0.0), (15.0, 0.0)],
        ]
        cones, colors = create_simple_cones()
        coordinate_confidence = np.zeros(len(cones))

        result = beam_search_prune(paths, cones, coordinate_confidence, colors, beam_width=2)

        assert isinstance(result, list)

    def test_returns_correct_beam_width(self):
        """Should return at most beam_width paths."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0)],
            [(0.0, 0.0), (5.0, 0.5), (10.0, 0.0), (15.0, 0.0)],
            [(0.0, 0.0), (5.0, -0.5), (10.0, 0.0), (15.0, 0.0)],
            [(0.0, 0.0), (5.0, 1.0), (10.0, 0.0), (15.0, 0.0)],
        ]
        cones, colors = create_simple_cones()
        coordinate_confidence = np.zeros(len(cones))

        result = beam_search_prune(paths, cones, coordinate_confidence, colors, beam_width=2)

        assert len(result) <= 2

    def test_empty_paths_returns_empty(self):
        """Should return empty list for empty input."""
        paths = []
        cones, colors = create_simple_cones()
        coordinate_confidence = np.zeros(len(cones))

        result = beam_search_prune(paths, cones, coordinate_confidence, colors, beam_width=5)

        assert result == []

    def test_preserves_path_structure(self):
        """Output paths should have same structure as input."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0)],
        ]
        cones, colors = create_simple_cones()
        coordinate_confidence = np.zeros(len(cones))

        result = beam_search_prune(paths, cones, coordinate_confidence, colors, beam_width=1)

        assert len(result) == 1
        assert result[0] == paths[0]


class TestBeamSearchWithDifferentBeamWidths:
    """Tests for different beam width values."""

    def test_beam_width_1(self):
        """Should return only the best path."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0)],
            [(0.0, 0.0), (5.0, 1.0), (10.0, 0.0), (15.0, 0.0)],
            [(0.0, 0.0), (5.0, 2.0), (10.0, 0.0), (15.0, 0.0)],
        ]
        cones, colors = create_simple_cones()
        coordinate_confidence = np.zeros(len(cones))

        result = beam_search_prune(paths, cones, coordinate_confidence, colors, beam_width=1)

        assert len(result) == 1

    def test_beam_width_larger_than_paths(self):
        """Should return all paths when beam_width > len(paths)."""
        paths = [
            [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0)],
            [(0.0, 0.0), (5.0, 0.5), (10.0, 0.0), (15.0, 0.0)],
        ]
        cones, colors = create_simple_cones()
        coordinate_confidence = np.zeros(len(cones))

        result = beam_search_prune(paths, cones, coordinate_confidence, colors, beam_width=10)

        assert len(result) == 2
