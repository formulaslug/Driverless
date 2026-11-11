"""Autonomous Inspection Mission implementation with NumPy test profiles."""

from enum import Enum
from typing import Optional
import numpy as np

from Pipeline.messages import Mission, ActuationCommandMessage
from StateMachine.level2.base import MissionBehavior, MissionContext, SubStateResult
from StateMachine.level2.registry import MissionRegistry


class InspectionSubState(str, Enum):
    """Sub-states of the Autonomous Inspection Mission."""
    STEERING_SWEEP = "STEERING_SWEEP"
    POWERTRAIN_RAMP = "POWERTRAIN_RAMP"
    CONTROLLED_STOP = "CONTROLLED_STOP"
    COMPLETE = "COMPLETE"


class InspectionMission(MissionBehavior):
    """Minimal baseline MVP inspection mission.
    
    Generates a deterministic, open-loop actuator validation profile:
    1. STEERING_SWEEP: Sinusoidal steering sweep at standstill.
    2. POWERTRAIN_RAMP: Low-speed throttle ramp and hold.
    3. CONTROLLED_STOP: Service braking until standstill is achieved.
    4. COMPLETE: Signals mission finished to Level 1.
    """

    # Sub-state duration limits
    STEER_SWEEP_DURATION_S: float = 4.0
    POWERTRAIN_DURATION_S: float = 4.0
    BRAKING_TIMEOUT_S: float = 3.0

    # Sinusoidal steering parameters
    STEER_AMPLITUDE_RAD: float = 0.25   # ~14.3 degrees
    STEER_FREQUENCY_HZ: float = 0.5

    # Low-speed throttle parameters
    THROTTLE_PEAK_NORM: float = 0.15
    THROTTLE_RAMP_UP_S: float = 1.5
    THROTTLE_RAMP_DOWN_S: float = 1.0

    # Braking command
    BRAKE_STOP_NORM: float = 0.8
    STANDSTILL_THRESHOLD_MPS: float = 0.05

    def __init__(self) -> None:
        self._substate: InspectionSubState = InspectionSubState.STEERING_SWEEP
        self._substate_start_time: float = 0.0

    @property
    def name(self) -> str:
        return "Autonomous Inspection Mission"

    @property
    def current_substate(self) -> InspectionSubState:
        return self._substate

    def on_enter(self, context: MissionContext) -> None:
        """Initialize inspection routine upon transition to AS_Driving."""
        self._substate = InspectionSubState.STEERING_SWEEP
        self._substate_start_time = context.mission_time

    def update(self, context: MissionContext) -> SubStateResult:
        """Compute open-loop test commands for the current tick."""
        t_sub = context.mission_time - self._substate_start_time

        # Check sub-state transitions prior to command computation
        if self._substate == InspectionSubState.STEERING_SWEEP:
            if t_sub >= self.STEER_SWEEP_DURATION_S:
                self._substate = InspectionSubState.POWERTRAIN_RAMP
                self._substate_start_time = context.mission_time
                t_sub = 0.0

        elif self._substate == InspectionSubState.POWERTRAIN_RAMP:
            if t_sub >= self.POWERTRAIN_DURATION_S:
                self._substate = InspectionSubState.CONTROLLED_STOP
                self._substate_start_time = context.mission_time
                t_sub = 0.0

        elif self._substate == InspectionSubState.CONTROLLED_STOP:
            v_x = float(context.telemetry.velocity[0])
            is_still = abs(v_x) < self.STANDSTILL_THRESHOLD_MPS
            if (is_still and t_sub >= 0.5) or (t_sub >= self.BRAKING_TIMEOUT_S):
                self._substate = InspectionSubState.COMPLETE
                self._substate_start_time = context.mission_time
                t_sub = 0.0

        steering = np.zeros(1, dtype=np.float64)
        throttle = np.zeros(1, dtype=np.float64)
        braking = np.zeros(1, dtype=np.float64)
        mission_finished = False

        if self._substate == InspectionSubState.STEERING_SWEEP:
            # Generate sinusoidal sweep: A * sin(2 * pi * f * t)
            steering[0] = self.STEER_AMPLITUDE_RAD * np.sin(
                2.0 * np.pi * self.STEER_FREQUENCY_HZ * t_sub
            )
            throttle[0] = 0.0
            braking[0] = 0.0

        elif self._substate == InspectionSubState.POWERTRAIN_RAMP:
            # Trapezoidal throttle ramp
            steering[0] = 0.0
            braking[0] = 0.0

            if t_sub < self.THROTTLE_RAMP_UP_S:
                throttle[0] = self.THROTTLE_PEAK_NORM * (t_sub / self.THROTTLE_RAMP_UP_S)
            elif t_sub < (self.POWERTRAIN_DURATION_S - self.THROTTLE_RAMP_DOWN_S):
                throttle[0] = self.THROTTLE_PEAK_NORM
            else:
                remaining = self.POWERTRAIN_DURATION_S - t_sub
                throttle[0] = np.maximum(
                    0.0,
                    self.THROTTLE_PEAK_NORM * (remaining / self.THROTTLE_RAMP_DOWN_S),
                )

        elif self._substate == InspectionSubState.CONTROLLED_STOP:
            # Controlled stop to standstill
            steering[0] = 0.0
            throttle[0] = 0.0
            braking[0] = self.BRAKE_STOP_NORM

        elif self._substate == InspectionSubState.COMPLETE:
            steering[0] = 0.0
            throttle[0] = 0.0
            braking[0] = 1.0
            mission_finished = True

        cmd = ActuationCommandMessage(
            timestamp=context.telemetry.timestamp,
            steering=steering,
            throttle=throttle,
            braking=braking,
            gated=True,
        )

        return SubStateResult(
            next_substate=self._substate.value,
            mission_finished=mission_finished,
            emergency_stop=False,
            actuation_command=cmd,
        )

    def on_exit(self, context: MissionContext, interrupted: bool = False) -> None:
        """Cleanup upon exiting inspection driving."""
        self._substate = InspectionSubState.COMPLETE


# Auto-register with MissionRegistry
MissionRegistry.register(Mission.INSPECTION, InspectionMission)
