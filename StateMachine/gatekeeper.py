"""Level 3 Actuation Safety Multiplexer and Gatekeeper."""

from typing import Optional
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Pipeline'))

from messages import MacroState, ActuationCommandMessage
from StateMachine.constants import (
    BRAKE_SERVICE_NORM,
    BRAKE_EBS_NORM,
    BRAKE_ZERO_NORM,
    THROTTLE_ZERO_NORM,
    STEERING_ZERO_RAD,
)


class ActuationGatekeeper:
    """Safety multiplexer enforcing Level 1 actuation gating policies.
    
    Acts as the single point of control for outgoing actuator messages:
    - In AS_Driving: Passes through requested steering, throttle, and braking arrays.
    - In AS_Ready: Zeroes throttle and steering; commands full service brake.
    - In AS_Finished: Zeroes throttle and steering; commands full EBS braking.
    - In AS_Emergency: Zeroes throttle and steering; commands full EBS braking.
    - In AS_Off / Manual: Zeroes throttle and steering; releases brakes.
    """

    @staticmethod
    def gate_command(
        state: MacroState,
        requested_command: Optional[ActuationCommandMessage],
        timestamp: float,
    ) -> ActuationCommandMessage:
        """Filter requested actuation commands according to the active macro-state.
        
        Args:
            state: Current Level 1 MacroState.
            requested_command: Raw actuation command requested by Level 2 / Controls.
            timestamp: Current monotonic evaluation timestamp.
            
        Returns:
            Gated ActuationCommandMessage safe for transmission to the VCU bridge.
        """
        # In AS_Driving, pass through valid Level 2 commands
        if state == MacroState.AS_DRIVING and requested_command is not None:
            steering = np.asarray(requested_command.steering, dtype=np.float64)
            throttle = np.asarray(requested_command.throttle, dtype=np.float64)
            braking = np.asarray(requested_command.braking, dtype=np.float64)
            return ActuationCommandMessage(
                timestamp=timestamp,
                steering=steering,
                throttle=throttle,
                braking=braking,
                gated=True,
            )

        # In all other states, determine safe default command arrays
        shape = (1,)
        if requested_command is not None and isinstance(requested_command.steering, np.ndarray):
            shape = requested_command.steering.shape

        steering = np.full(shape, STEERING_ZERO_RAD, dtype=np.float64)
        throttle = np.full(shape, THROTTLE_ZERO_NORM, dtype=np.float64)

        if state == MacroState.AS_READY:
            braking = np.full(shape, BRAKE_SERVICE_NORM, dtype=np.float64)
        elif state in (MacroState.AS_FINISHED, MacroState.AS_EMERGENCY):
            braking = np.full(shape, BRAKE_EBS_NORM, dtype=np.float64)
        else:
            braking = np.full(shape, BRAKE_ZERO_NORM, dtype=np.float64)

        return ActuationCommandMessage(
            timestamp=timestamp,
            steering=steering,
            throttle=throttle,
            braking=braking,
            gated=False,
        )
