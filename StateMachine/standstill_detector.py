"""NumPy-accelerated standstill detection via rolling ring buffer."""

from typing import Union
import numpy as np

from StateMachine.constants import (
    STANDSTILL_VELOCITY_THRESHOLD_MPS,
    STANDSTILL_MIN_DURATION_S,
    STANDSTILL_BUFFER_CAPACITY,
)


class StandstillDetector:
    """Tracks longitudinal velocity history in a circular ring buffer.

    Checks that the vehicle has stayed within the standstill deadband
    (|v_x| < threshold) for at least a given duration. Each push is O(1) and
    never reallocates memory.

    Typical usage example:

        detector = StandstillDetector()
        detector.push(timestamp=t, velocity=v_x)
        if detector.is_standstill():
            ...

    Args:
        capacity: Maximum number of samples to keep. Once full, the oldest
            sample is overwritten.

    Attributes:
        capacity: Maximum number of buffered samples.
        sample_count: Number of valid samples currently buffered.

    Raises:
        ValueError: If capacity is not strictly positive.
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
        """Appends a velocity sample, overwriting the oldest one when full.

        Args:
            timestamp: Sample time in seconds.
            velocity: Longitudinal velocity v_x in m/s. If an array is given,
                only its first element is used.
        """
        v_arr = np.atleast_1d(velocity)
        v_x = float(v_arr[0])

        self._buffer[self._head, 0] = timestamp
        self._buffer[self._head, 1] = v_x
        self._head = (self._head + 1) % self.capacity

        if self._count < self.capacity:
            self._count += 1

    def get_window(self) -> np.ndarray:
        """Returns the buffered samples in chronological order.

        Returns:
            A copy of the samples as an (N, 2) array, oldest first, with
            columns [timestamp, v_x]. Empty with shape (0, 2) if nothing is
            buffered.
        """
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
        """Checks whether the vehicle has been at standstill long enough.

        The samples in the last required_duration_s must span at least 90% of
        that duration, which allows for small gaps between samples, and every
        one must satisfy |v_x| < threshold_mps.

        Args:
            required_duration_s: Time in seconds the vehicle must stay at
                standstill.
            threshold_mps: Speed in m/s below which the vehicle counts as
                stationary.

        Returns:
            True if the standstill criteria have been met continuously for
            the required duration, otherwise False. Always False with fewer
            than two samples.
        """
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
        """Clears all buffered samples."""
        self._buffer.fill(0.0)
        self._count = 0
        self._head = 0

    @property
    def sample_count(self) -> int:
        """The number of valid samples currently buffered."""
        return self._count
