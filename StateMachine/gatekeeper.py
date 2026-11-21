"""Level 3 actuation safety gate.

Every outgoing actuator command passes through ``gate_command``:
- DS_Driving: a valid command is clamped to physical limits and passed through
  (``gated=True``). A missing, non-finite or mis-shaped command is replaced by a
  service-brake stop.
- Every other state: throttle and steering are forced to zero and the brake is
  set by the state (service brake in DS_Ready, EBS in DS_Finished/DS_Emergency,
  released in DS_Off/Manual).
"""

from typing import Optional, Tuple

import numpy as np

from Pipeline.messages import MacroState, ActuationCommandMessage
from StateMachine.constants import (
    BRAKE_SERVICE_NORM,
    BRAKE_EBS_NORM,
    BRAKE_ZERO_NORM,
    THROTTLE_ZERO_NORM,
    STEERING_ZERO_RAD,
    THROTTLE_MIN_NORM,
    THROTTLE_MAX_NORM,
    BRAKE_MIN_NORM,
    BRAKE_MAX_NORM,
    STEERING_MAX_RAD,
)

# Brake held in each state when actuation is not passed through.
_HOLD_BRAKE = {
    MacroState.DS_READY: BRAKE_SERVICE_NORM,
    MacroState.DS_DRIVING: BRAKE_SERVICE_NORM,
    MacroState.DS_FINISHED: BRAKE_EBS_NORM,
    MacroState.DS_EMERGENCY: BRAKE_EBS_NORM,
}


def _stop_command(timestamp: float, shape: Tuple[int, ...], braking: float) -> ActuationCommandMessage:
    """Builds a command with zero steering and throttle and the given brake.

    Args:
        timestamp: Current telemetry timestamp.
        shape: Shape shared by the steering, throttle and braking arrays.
        braking: Normalized brake command applied to every element.

    Returns:
        An ActuationCommandMessage with gated=False, since the command was
        generated here rather than passed through from a planner.
    """
    return ActuationCommandMessage(
        timestamp=timestamp,
        steering=np.full(shape, STEERING_ZERO_RAD),
        throttle=np.full(shape, THROTTLE_ZERO_NORM),
        braking=np.full(shape, braking),
        gated=False,
    )


def _is_valid(cmd: ActuationCommandMessage) -> bool:
    """Checks that a command is safe to clamp and pass through.

    Args:
        cmd: Actuation command requested by the planner.

    Returns:
        True if the steering, throttle and braking arrays all have the same
        shape and contain only finite values, otherwise False.
    """
    arrays = (cmd.steering, cmd.throttle, cmd.braking)
    same_shape = cmd.steering.shape == cmd.throttle.shape == cmd.braking.shape
    return same_shape and all(np.isfinite(a).all() for a in arrays)


def gate_command(
    state: MacroState,
    requested_command: Optional[ActuationCommandMessage],
    timestamp: float,
) -> ActuationCommandMessage:
    """Returns the actuation command that is safe to send to the VCU.

    In DS_DRIVING, a valid command is clamped to physical limits and passed
    through. A missing or invalid one is replaced by a service-brake stop. In
    every other state the requested command is ignored: steering and throttle
    are zero and the brake is set by the state.

    Args:
        state: Current macro-state.
        requested_command: Command requested by the planner, or None if there
            is none.
        timestamp: Current telemetry timestamp.

    Returns:
        An ActuationCommandMessage with gated=True if the requested command
        was passed through, otherwise gated=False.
    """
    shape = requested_command.steering.shape if requested_command is not None else (1,)

    if state != MacroState.DS_DRIVING or requested_command is None:
        return _stop_command(timestamp, shape, _HOLD_BRAKE.get(state, BRAKE_ZERO_NORM))

    if not _is_valid(requested_command):
        return _stop_command(timestamp, shape, BRAKE_SERVICE_NORM)

    return ActuationCommandMessage(
        timestamp=timestamp,
        steering=np.clip(requested_command.steering, -STEERING_MAX_RAD, STEERING_MAX_RAD),
        throttle=np.clip(requested_command.throttle, THROTTLE_MIN_NORM, THROTTLE_MAX_NORM),
        braking=np.clip(requested_command.braking, BRAKE_MIN_NORM, BRAKE_MAX_NORM),
        gated=True,
    )
