"""Level 3 Actuation Safety Multiplexer and Gatekeeper."""

from typing import Optional
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


class ActuationGatekeeper:
    """Safety multiplexer enforcing Level 1 actuation gating policies.
    
    Acts as the single point of control for outgoing actuator messages:
    - In AS_Driving: Sanitizes, validates (NaN/Inf check), and clamps requested commands.
      If no command is provided, clamps fail-safe service brake.
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
        """Filter and sanitize requested actuation commands according to the active macro-state.
        
        Args:
            state: Current Level 1 MacroState.
            requested_command: Raw actuation command requested by Level 2 / Controls.
            timestamp: Current monotonic evaluation timestamp.
            
        Returns:
            Gated ActuationCommandMessage safe for transmission to the VCU bridge.
        """
        # Determine command array shape for zeroed fallbacks
        shape = (1,)
        if requested_command is not None and isinstance(requested_command.steering, np.ndarray):
            shape = requested_command.steering.shape

        # In AS_Driving, evaluate requested commands
        if state == MacroState.AS_DRIVING:
            if requested_command is not None:
                steering = np.asarray(requested_command.steering, dtype=np.float64)
                throttle = np.asarray(requested_command.throttle, dtype=np.float64)
                braking = np.asarray(requested_command.braking, dtype=np.float64)

                # Reject corrupted or unbounded commands (NaN / Inf protection)
                has_invalid_floats = (
                    np.any(np.isnan(steering))
                    or np.any(np.isnan(throttle))
                    or np.any(np.isnan(braking))
                    or np.any(np.isinf(steering))
                    or np.any(np.isinf(throttle))
                    or np.any(np.isinf(braking))
                )
                if has_invalid_floats:
                    return ActuationCommandMessage(
                        timestamp=timestamp,
                        steering=np.full(shape, STEERING_ZERO_RAD, dtype=np.float64),
                        throttle=np.full(shape, THROTTLE_ZERO_NORM, dtype=np.float64),
                        braking=np.full(shape, BRAKE_SERVICE_NORM, dtype=np.float64),
                        gated=False,
                    )

                # Clamp valid commands to safe physical and normalized bounds
                steering = np.clip(steering, -STEERING_MAX_RAD, STEERING_MAX_RAD)
                throttle = np.clip(throttle, THROTTLE_MIN_NORM, THROTTLE_MAX_NORM)
                braking = np.clip(braking, BRAKE_MIN_NORM, BRAKE_MAX_NORM)

                return ActuationCommandMessage(
                    timestamp=timestamp,
                    steering=steering,
                    throttle=throttle,
                    braking=braking,
                    gated=True,
                )
            else:
                # Safe fallback in driving when no command is available: clamp service brake
                return ActuationCommandMessage(
                    timestamp=timestamp,
                    steering=np.full(shape, STEERING_ZERO_RAD, dtype=np.float64),
                    throttle=np.full(shape, THROTTLE_ZERO_NORM, dtype=np.float64),
                    braking=np.full(shape, BRAKE_SERVICE_NORM, dtype=np.float64),
                    gated=False,
                )

        # In all non-driving states, enforce zeroed throttle and steering
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
