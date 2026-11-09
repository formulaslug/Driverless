"""NumPy-accelerated standstill detection via rolling window buffer."""

from typing import Union
import numpy as np

from StateMachine.constants import (
    STANDSTILL_VELOCITY_THRESHOLD_MPS,
    STANDSTILL_MIN_DURATION_S,
    STANDSTILL_BUFFER_CAPACITY,
)


class StandstillDetector:
    """Tracks vehicle longitudinal velocity history in a circular NumPy buffer.
    
    Determines whether the vehicle has remained continuously within a standstill
    deadband (|v_x| < threshold) for at least a specified time duration.
    """

    def __init__(self, capacity: int = STANDSTILL_BUFFER_CAPACITY) -> None:
        """Initialize circular buffer.
        
        Args:
            capacity: Maximum number of samples to store in ring buffer.
        """
        if capacity <= 0:
            raise ValueError("Buffer capacity must be strictly positive.")
        self.capacity: int = capacity
        # Buffer columns: [0: timestamp, 1: longitudinal_velocity_vx]
        self._buffer: np.ndarray = np.zeros((capacity, 2), dtype=np.float64)
        self._count: int = 0
        self._head: int = 0

    def push(self, timestamp: float, velocity: Union[float, np.ndarray]) -> None:
        """Append a velocity observation to the buffer.
        
        Args:
            timestamp: Monotonic sample timestamp in seconds.
            velocity: Longitudinal velocity float or vector whose first component is v_x.
        """
        if isinstance(velocity, np.ndarray):
            v_x = float(velocity[0])
        else:
            v_x = float(velocity)

        self._buffer[self._head, 0] = timestamp
        self._buffer[self._head, 1] = v_x
        self._head = (self._head + 1) % self.capacity
        if self._count < self.capacity:
            self._count += 1

    def get_window(self) -> np.ndarray:
        """Return all valid buffer entries in chronological order.
        
        Returns:
            np.ndarray of shape (N, 2) containing [timestamp, v_x].
        """
        if self._count == 0:
            return np.empty((0, 2), dtype=np.float64)

        if self._count < self.capacity:
            return self._buffer[:self._count].copy()

        # Unroll ring buffer chronologically
        return np.concatenate((self._buffer[self._head:], self._buffer[:self._head]), axis=0)

    def is_standstill(
        self,
        required_duration_s: float = STANDSTILL_MIN_DURATION_S,
        threshold_mps: float = STANDSTILL_VELOCITY_THRESHOLD_MPS,
    ) -> bool:
        """Evaluate if vehicle has continuously satisfied standstill criteria.
        
        Args:
            required_duration_s: Minimum duration vehicle must remain within deadband.
            threshold_mps: Velocity magnitude limit for standstill (|v_x| < threshold).
            
        Returns:
            True if buffer confirms continuous standstill over required duration, False otherwise.
        """
        if self._count < 2:
            return False

        window = self.get_window()
        t_latest = window[-1, 0]
        t_cutoff = t_latest - required_duration_s

        # Mask points within the required duration window
        time_mask = window[:, 0] >= t_cutoff
        sub_window = window[time_mask]

        if sub_window.shape[0] < 2:
            return False

        # Ensure the window actually spans the required duration (allowing small dt tolerance)
        actual_span = t_latest - sub_window[0, 0]
        if actual_span < (required_duration_s * 0.90):
            return False

        # Vectorized check: all longitudinal velocities within threshold
        velocities = sub_window[:, 1]
        within_deadband = np.abs(velocities) < threshold_mps

        return bool(np.all(within_deadband))

    def reset(self) -> None:
        """Clear all buffered observations."""
        self._buffer.fill(0.0)
        self._count = 0
        self._head = 0

    @property
    def sample_count(self) -> int:
        """Number of valid samples currently buffered."""
        return self._count
