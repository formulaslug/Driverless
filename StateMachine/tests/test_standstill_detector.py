"""Unit tests for the NumPy rolling-window standstill detector."""

import pytest
import numpy as np

from StateMachine.standstill_detector import StandstillDetector


def test_empty_buffer_returns_false() -> None:
    """Empty buffer does not report standstill."""
    detector = StandstillDetector(capacity=50)
    assert not detector.is_standstill()
    assert detector.sample_count == 0


def test_insufficient_time_span_returns_false() -> None:
    """Zero velocity over insufficient duration (< 0.5s) does not report standstill."""
    detector = StandstillDetector(capacity=50)
    detector.push(timestamp=1.0, velocity=0.0)
    detector.push(timestamp=1.1, velocity=0.0)
    detector.push(timestamp=1.2, velocity=0.0)  # Only 0.2s duration

    assert not detector.is_standstill(required_duration_s=0.5)


def test_continuous_standstill_satisfies_criteria() -> None:
    """Longitudinal velocity consistently < 0.05 m/s over 0.5s reports standstill."""
    detector = StandstillDetector(capacity=50)
    t = 0.0
    for _ in range(30):  # 30 * 0.02s = 0.6s
        detector.push(timestamp=t, velocity=0.01)
        t += 0.02

    assert detector.is_standstill(required_duration_s=0.5, threshold_mps=0.05)


def test_velocity_exceeding_threshold_rejects_standstill() -> None:
    """Velocity >= 0.05 m/s rejects standstill even if constant."""
    detector = StandstillDetector(capacity=50)
    t = 0.0
    for _ in range(30):
        detector.push(timestamp=t, velocity=0.08)
        t += 0.02

    assert not detector.is_standstill(required_duration_s=0.5, threshold_mps=0.05)


def test_spike_in_window_invalidates_standstill() -> None:
    """A transient velocity spike within the required window invalidates standstill."""
    detector = StandstillDetector(capacity=50)
    t = 0.0
    for i in range(30):
        # Spike at sample 20 (0.4s into window)
        v = 0.12 if i == 20 else 0.01
        detector.push(timestamp=t, velocity=v)
        t += 0.02

    assert not detector.is_standstill(required_duration_s=0.5, threshold_mps=0.05)


def test_circular_buffer_rollover() -> None:
    """Ring buffer correctly handles rollover with capacity = 20 and 50 samples pushed."""
    detector = StandstillDetector(capacity=20)
    t = 0.0
    # First push 30 moving samples
    for _ in range(30):
        detector.push(timestamp=t, velocity=5.0)
        t += 0.02

    # Now push 20 standstill samples (0.4s)
    for _ in range(20):
        detector.push(timestamp=t, velocity=0.005)
        t += 0.02

    window = detector.get_window()
    assert window.shape == (20, 2)
    # Check that time is monotonically increasing in unrolled window
    assert np.all(np.diff(window[:, 0]) > 0)


def test_numpy_vector_input() -> None:
    """Buffer accepts multi-dimensional velocity vectors and slices longitudinal v_x."""
    detector = StandstillDetector(capacity=50)
    t = 0.0
    for _ in range(30):
        # [v_x, v_y, yaw_rate]
        vec = np.array([0.02, 1.5, 0.4], dtype=np.float64)
        detector.push(timestamp=t, velocity=vec)
        t += 0.02

    assert detector.is_standstill(required_duration_s=0.5, threshold_mps=0.05)


def test_numpy_0d_scalar_input() -> None:
    """Buffer safely handles 0-dimensional scalar NumPy arrays without indexing errors."""
    detector = StandstillDetector(capacity=50)
    t = 0.0
    for _ in range(30):
        scalar_arr = np.array(0.01, dtype=np.float64)
        detector.push(timestamp=t, velocity=scalar_arr)
        t += 0.02

    assert detector.is_standstill(required_duration_s=0.5, threshold_mps=0.05)


def test_reset_clears_buffer() -> None:
    """Reset clears sample count and buffer contents."""
    detector = StandstillDetector(capacity=50)
    detector.push(timestamp=1.0, velocity=0.0)
    detector.reset()
    assert detector.sample_count == 0
    assert not detector.is_standstill()
