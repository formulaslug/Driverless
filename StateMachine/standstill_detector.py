"""NumPy-accelerated standstill detection via rolling ring buffer."""

from typing import Union
import numpy as np

from StateMachine.constants import (
    STANDSTILL_VELOCITY_THRESHOLD_MPS,
    STANDSTILL_MIN_DURATION_S,
    STANDSTILL_BUFFER_CAPACITY,
)


class StandstillDetector:
    """Tracks vehicle longitudinal velocity history in a circular ring buffer.
    
    Verifies that the vehicle has remained continuously within the standstill
    deadband (|v_x| < threshold) for at least a specified time duration.
    Operates in O(1) time per push without periodic memory reallocations.
    """

    def __init__(self, capacity: int = STANDSTILL_BUFFER_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("Buffer capacity must be strictly positive.")
        self.capacity: int = capacity
        # Buffer columns: [0: timestamp, 1: longitudinal_velocity_vx]
        self._buffer: np.ndarray = np.zeros((capacity, 2), dtype=np.float64)
        self._count: int = 0
        self._head: int = 0

    def push(self, timestamp: float, velocity: Union[float, np.ndarray]) -> None:
        """Append a velocity observation in O(1) time using circular indexing."""
        v_x = float(velocity[0]) if isinstance(velocity, np.ndarray) else float(velocity)

        self._buffer[self._head, 0] = timestamp
        self._buffer[self._head, 1] = v_x
        self._head = (self._head + 1) % self.capacity

        if self._count < self.capacity:
            self._count += 1

    def get_window(self) -> np.ndarray:
        """Return chronological history of valid buffered samples."""
        if self._count == 0:
            return np.empty((0, 2), dtype=np.float64)

        if self._count < self.capacity:
            return self._buffer[:self._count].copy()

        # Unroll ring buffer chronologically from oldest to newest sample
        return np.concatenate((self._buffer[self._head:], self._buffer[:self._head]), axis=0)

    def is_standstill(
        self,
        required_duration_s: float = STANDSTILL_MIN_DURATION_S,
        threshold_mps: float = STANDSTILL_VELOCITY_THRESHOLD_MPS,
    ) -> bool:
        """Return True if vehicle has continuously satisfied standstill criteria."""
        if self._count < 2:
            return False

        window = self.get_window()
        t_latest = window[-1, 0]
        cutoff = t_latest - required_duration_s

        # Extract samples within the duration window
        recent = window[window[:, 0] >= cutoff]
        if recent.shape[0] < 2:
            return False

        # Ensure the samples span the required duration (allowing small dt tolerance)
        actual_span = t_latest - recent[0, 0]
        if actual_span < (required_duration_s * 0.90):
            return False

        # Vectorized check: all velocities within standstill deadband
        return bool(np.all(np.abs(recent[:, 1]) < threshold_mps))

    def reset(self) -> None:
        """Clear all buffered observations."""
        self._buffer.fill(0.0)
        self._count = 0
        self._head = 0

    @property
    def sample_count(self) -> int:
        """Number of valid samples currently buffered."""
        return self._count
