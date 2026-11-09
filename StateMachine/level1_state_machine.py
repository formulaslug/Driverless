"""Deterministic Level 1 Macro-State Machine for FSAE Driverless."""

from typing import Optional, Tuple, Set
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Pipeline'))

from messages import (
    MacroState,
    ASSIStatus,
    EbsStatus,
    Mission,
    DiagnosticErrorCode,
    VcuTelemetryMessage,
    ActuationCommandMessage,
    StateMachineStatusMessage,
)
from StateMachine.constants import (
    AS_READY_LOCKOUT_DURATION_S,
    VCU_WATCHDOG_TIMEOUT_S,
    STANDSTILL_MIN_DURATION_S,
    STANDSTILL_VELOCITY_THRESHOLD_MPS,
    HEARTBEAT_MAX,
)
from StateMachine.standstill_detector import StandstillDetector
from StateMachine.gatekeeper import ActuationGatekeeper
from StateMachine.level2.base import MissionBehavior, MissionContext, SubStateResult
from StateMachine.level2.registry import MissionRegistry


VALID_DRIVERLESS_MISSIONS: Set[Mission] = {
    Mission.INSPECTION,
    Mission.ACCELERATION,
    Mission.SKIDPAD,
    Mission.AUTOCROSS,
    Mission.TRACKDRIVE,
}

_NUM_DIAGNOSTIC_BITS: int = 16
_BIT_INDICES: np.ndarray = np.arange(_NUM_DIAGNOSTIC_BITS, dtype=np.int32)


def unpack_diagnostic_flags(error_code: int) -> np.ndarray:
    """Vectorized unpacking of integer error bitmask into a NumPy boolean array.
    
    Args:
        error_code: Integer diagnostic bitmask.
        
    Returns:
        np.ndarray of shape (16,) containing boolean status for each bit.
    """
    return ((error_code >> _BIT_INDICES) & 1).astype(bool)


class Level1StateMachine:
    """FSAE Driverless Level 1 Macro-State Machine.
    
    Governs the top-level safety states:
    - AS_Off: System quiescent, checks readiness preconditions.
    - AS_Ready: Preconditions verified, 5.0s lockout enforced, service brake engaged.
    - AS_Driving: Autonomous mission active, 50 Hz watchdogs, Level 3 actuation multiplexed.
    - AS_Finished: Autonomous run complete, verified standstill, EBS engaged.
    - AS_Emergency: Safety shutdown circuit open or fault tripped; latching.
    - Manual_Driving: TS active with manual mission and ASMS off.
    """

    def __init__(self) -> None:
        self._state: MacroState = MacroState.AS_OFF
        self._assi_status: ASSIStatus = ASSIStatus.OFF
        self._error_code: int = DiagnosticErrorCode.NONE

        # Timers and timestamps
        self._state_entry_time: float = 0.0
        self._ready_start_time: float = 0.0
        self._last_vcu_telemetry_time: Optional[float] = None
        self._last_tick_time: Optional[float] = None

        # Lockout and RES Go handling
        self._premature_res_detected: bool = False

        # Active mission behavior and sub-state
        self._active_mission_behavior: Optional[MissionBehavior] = None
        self._active_substate_name: str = "NONE"
        self._mission_start_time: float = 0.0

        # Commands and outputs
        self._heartbeat: int = 0
        self._ebs_trigger_cmd: bool = False
        self._service_brake_cmd: bool = False

        # Standstill detection buffer
        self._standstill_detector: StandstillDetector = StandstillDetector()

    @property
    def current_state(self) -> MacroState:
        """Active Level 1 MacroState."""
        return self._state

    @property
    def assi_status(self) -> ASSIStatus:
        """Active Autonomous System Status Indicator state."""
        return self._assi_status

    @property
    def error_code(self) -> int:
        """Current diagnostic error code bitmask."""
        return self._error_code

    @property
    def heartbeat(self) -> int:
        """Rolling heartbeat counter."""
        return self._heartbeat

    @property
    def ready_timer_elapsed(self) -> float:
        """Elapsed time in AS_Ready state (seconds)."""
        if self._state == MacroState.AS_READY:
            return max(0.0, (self._last_tick_time or 0.0) - self._ready_start_time)
        return 0.0

    @property
    def active_mission(self) -> Optional[Mission]:
        """Currently active driverless mission."""
        if self._active_mission_behavior is not None:
            # Look up registered mission
            return getattr(self._active_mission_behavior, "_mission_enum", None)
        return None

    @property
    def active_substate_name(self) -> str:
        """Current Level 2 sub-state name."""
        return self._active_substate_name

    def request_emergency_trip(self, fault: DiagnosticErrorCode, timestamp: float) -> None:
        """Explicitly trip the system into AS_Emergency with diagnostic fault code.
        
        Args:
            fault: Diagnostic fault bitmask.
            timestamp: Current monotonic timestamp.
        """
        self._transition_to_emergency(fault, timestamp)

    def request_reset(self, telemetry: VcuTelemetryMessage) -> bool:
        """Attempt to clear AS_Emergency latch and return to AS_Off.
        
        Per rulebook / flowchart: Requires EBS sound off (EBS unlatched/deactivated),
        ASMS switch off, and brakes released.
        
        Args:
            telemetry: Current VCU telemetry.
            
        Returns:
            True if reset succeeded and state transitioned to AS_Off; False if conditions not met.
        """
        if self._state != MacroState.AS_EMERGENCY:
            return False

        # Reset conditions: EBS deactivated, ASMS off, brakes released, SDC closed
        reset_conditions = np.array([
            telemetry.ebs_status != EbsStatus.ACTIVATED,
            not telemetry.asms_status,
            not telemetry.brake_status,
            telemetry.sdc_status,
        ], dtype=bool)

        if np.all(reset_conditions):
            self._error_code = DiagnosticErrorCode.NONE
            self._transition_to_state(MacroState.AS_OFF, telemetry.timestamp)
            return True

        return False

    def step(
        self,
        telemetry: VcuTelemetryMessage,
        external_actuation: Optional[ActuationCommandMessage] = None,
    ) -> Tuple[StateMachineStatusMessage, ActuationCommandMessage]:
        """Execute one deterministic cycle of the state machine.
        
        Args:
            telemetry: Fresh VCU telemetry message.
            external_actuation: Optional actuation command from external Controls pipeline.
            
        Returns:
            Tuple of (StateMachineStatusMessage, gated ActuationCommandMessage).
        """
        now = telemetry.timestamp
        dt = (now - self._last_tick_time) if self._last_tick_time is not None else 0.02
        self._last_tick_time = now
        self._heartbeat = (self._heartbeat + 1) & HEARTBEAT_MAX

        # Update standstill buffer with incoming velocity
        self._standstill_detector.push(now, telemetry.velocity)

        # Check VCU watchdog if we have prior telemetry
        if self._last_vcu_telemetry_time is not None:
            time_since_vcu = now - self._last_vcu_telemetry_time
            if time_since_vcu > VCU_WATCHDOG_TIMEOUT_S and self._state != MacroState.AS_OFF:
                self._transition_to_emergency(DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU, now)
        self._last_vcu_telemetry_time = now

        # Execute state-specific logic
        raw_actuation = external_actuation

        if self._state == MacroState.AS_OFF:
            raw_actuation = self._handle_as_off(telemetry, now)
        elif self._state == MacroState.AS_READY:
            raw_actuation = self._handle_as_ready(telemetry, now)
        elif self._state == MacroState.AS_DRIVING:
            raw_actuation = self._handle_as_driving(telemetry, dt, now, external_actuation)
        elif self._state == MacroState.AS_FINISHED:
            raw_actuation = self._handle_as_finished(telemetry, now)
        elif self._state == MacroState.AS_EMERGENCY:
            raw_actuation = self._handle_as_emergency(telemetry, now)
        elif self._state == MacroState.MANUAL_DRIVING:
            raw_actuation = self._handle_manual_driving(telemetry, now)

        # Apply Level 3 Actuation Gatekeeper
        gated_actuation = ActuationGatekeeper.gate_command(
            state=self._state,
            requested_command=raw_actuation,
            timestamp=now,
        )

        status_msg = StateMachineStatusMessage(
            timestamp=now,
            current_state=self._state,
            assi_status=self._assi_status,
            error_code=self._error_code,
            error_flags=unpack_diagnostic_flags(self._error_code),
            heartbeat=self._heartbeat,
            ebs_trigger_cmd=self._ebs_trigger_cmd,
            service_brake_cmd=self._service_brake_cmd,
            active_mission=telemetry.ami_mission,
            active_substate=self._active_substate_name,
            ready_timer_elapsed=self.ready_timer_elapsed,
        )

        return status_msg, gated_actuation

    # -----------------------------------------------------------------------
    # State Handlers
    # -----------------------------------------------------------------------

    def _handle_as_off(
        self, telemetry: VcuTelemetryMessage, now: float
    ) -> Optional[ActuationCommandMessage]:
        self._ebs_trigger_cmd = False
        self._service_brake_cmd = False

        # Check Manual Driving transition: Manual mission & EBS unavailable & ASMS off & TS active
        manual_conditions = np.array([
            telemetry.ami_mission == Mission.MANUAL,
            telemetry.ebs_status == EbsStatus.UNAVAILABLE,
            not telemetry.asms_status,
            telemetry.ts_status,
        ], dtype=bool)
        if np.all(manual_conditions):
            self._transition_to_state(MacroState.MANUAL_DRIVING, now)
            return None

        # Check AS_Off -> AS_Ready preconditions
        readiness_conditions = np.array([
            telemetry.ebs_status == EbsStatus.ARMED,
            telemetry.asms_status,
            telemetry.dbs_check_ok,
            telemetry.ts_status,
            telemetry.ami_mission in VALID_DRIVERLESS_MISSIONS,
            telemetry.brake_status,
            telemetry.sdc_status,
        ], dtype=bool)

        if np.all(readiness_conditions):
            self._transition_to_state(MacroState.AS_READY, now)

        return None

    def _handle_as_ready(
        self, telemetry: VcuTelemetryMessage, now: float
    ) -> Optional[ActuationCommandMessage]:
        # Enforce service brake engaged in AS_Ready
        self._service_brake_cmd = True
        self._ebs_trigger_cmd = False

        # Check critical emergency trips
        if not telemetry.sdc_status:
            self._transition_to_emergency(DiagnosticErrorCode.SDC_OPEN, now)
            return None
        if telemetry.ebs_status == EbsStatus.ACTIVATED:
            self._transition_to_emergency(DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED, now)
            return None

        # Check clean exit path to AS_Off: ASMS off & brakes released
        if (not telemetry.asms_status) and (not telemetry.brake_status):
            self._transition_to_state(MacroState.AS_OFF, now)
            return None

        elapsed_ready_time = now - self._ready_start_time

        # Enforce 5-second lockout timer:
        # Premature RES Go arrives before 5.0s -> latch premature flag to reject it
        if elapsed_ready_time < AS_READY_LOCKOUT_DURATION_S:
            if telemetry.res_go_signal:
                self._premature_res_detected = True
            return None

        # At t >= 5.0s:
        # If RES was pressed prematurely, operator must release it before a valid Go is accepted
        if self._premature_res_detected:
            if not telemetry.res_go_signal:
                # Signal released; now clear premature flag so next press is valid
                self._premature_res_detected = False
            return None

        # If fresh/valid RES Go signal is active and preconditions still hold -> AS_Driving
        if telemetry.res_go_signal:
            driving_preconditions = np.array([
                telemetry.ebs_status == EbsStatus.ARMED,
                telemetry.asms_status,
                telemetry.ts_status,
                telemetry.sdc_status,
                telemetry.dbs_check_ok,
            ], dtype=bool)

            if np.all(driving_preconditions):
                self._transition_to_driving(telemetry, now)

        return None

    def _handle_as_driving(
        self,
        telemetry: VcuTelemetryMessage,
        dt: float,
        now: float,
        external_actuation: Optional[ActuationCommandMessage],
    ) -> Optional[ActuationCommandMessage]:
        # Check critical emergency conditions
        if not telemetry.sdc_status:
            self._transition_to_emergency(DiagnosticErrorCode.SDC_OPEN, now)
            return None
        if telemetry.ebs_status == EbsStatus.ACTIVATED:
            self._transition_to_emergency(DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED, now)
            return None
        if not telemetry.asms_status:
            self._transition_to_emergency(DiagnosticErrorCode.ASMS_UNEXPECTED_DROP, now)
            return None
        if not telemetry.ts_status:
            self._transition_to_emergency(DiagnosticErrorCode.TS_INACTIVE, now)
            return None

        mission_time = now - self._mission_start_time
        context = MissionContext(
            telemetry=telemetry,
            dt=dt,
            mission_time=mission_time,
            current_substate=self._active_substate_name,
        )

        substate_result: Optional[SubStateResult] = None
        actuation_output: Optional[ActuationCommandMessage] = external_actuation

        if self._active_mission_behavior is not None:
            try:
                substate_result = self._active_mission_behavior.update(context)
                if substate_result.next_substate is not None:
                    self._active_substate_name = substate_result.next_substate
                if substate_result.actuation_command is not None:
                    actuation_output = substate_result.actuation_command
            except Exception:
                self._transition_to_emergency(DiagnosticErrorCode.LEVEL2_EXCEPTION, now)
                return None

        # Check if Level 2 requested an emergency stop
        if substate_result is not None and substate_result.emergency_stop:
            self._transition_to_emergency(DiagnosticErrorCode.LEVEL2_EXCEPTION, now)
            return None

        # Check mission completion and standstill verification
        level2_finished = (substate_result is not None and substate_result.mission_finished)
        if level2_finished:
            # Check continuous standstill over required duration via NumPy buffer
            is_still = self._standstill_detector.is_standstill(
                required_duration_s=STANDSTILL_MIN_DURATION_S,
                threshold_mps=STANDSTILL_VELOCITY_THRESHOLD_MPS,
            )
            if is_still:
                self._transition_to_state(MacroState.AS_FINISHED, now)
                return None

        return actuation_output

    def _handle_as_finished(
        self, telemetry: VcuTelemetryMessage, now: float
    ) -> Optional[ActuationCommandMessage]:
        # Engage EBS in AS_Finished per PDR & Flowchart
        self._ebs_trigger_cmd = True
        self._service_brake_cmd = False

        # Clean exit to AS_Off: ASMS off & brakes released
        if (not telemetry.asms_status) and (not telemetry.brake_status):
            self._transition_to_state(MacroState.AS_OFF, now)

        return None

    def _handle_as_emergency(
        self, telemetry: VcuTelemetryMessage, now: float
    ) -> Optional[ActuationCommandMessage]:
        # Engage full EBS in emergency
        self._ebs_trigger_cmd = True
        self._service_brake_cmd = True
        return None

    def _handle_manual_driving(
        self, telemetry: VcuTelemetryMessage, now: float
    ) -> Optional[ActuationCommandMessage]:
        self._ebs_trigger_cmd = False
        self._service_brake_cmd = False
        # Exit to AS_Off when TS is switched off
        if not telemetry.ts_status:
            self._transition_to_state(MacroState.AS_OFF, now)
        return None

    # -----------------------------------------------------------------------
    # Transition Helpers
    # -----------------------------------------------------------------------

    def _transition_to_state(self, new_state: MacroState, now: float) -> None:
        """Execute state entry/exit transitions."""
        old_state = self._state
        self._state = new_state
        self._state_entry_time = now

        # Map ASSI status
        if new_state == MacroState.AS_OFF:
            self._assi_status = ASSIStatus.OFF
            self._ebs_trigger_cmd = False
            self._service_brake_cmd = False
            self._active_substate_name = "NONE"
            self._premature_res_detected = False
            if self._active_mission_behavior is not None:
                self._active_mission_behavior.on_exit(
                    MissionContext(
                        telemetry=VcuTelemetryMessage(
                            timestamp=now,
                            ebs_status=EbsStatus.ARMED,
                            asms_status=False,
                            dbs_check_ok=True,
                            ts_status=False,
                            ami_mission=Mission.NOT_SELECTED,
                            brake_status=False,
                            sdc_status=True,
                            res_go_signal=False,
                        ),
                        dt=0.0,
                        mission_time=0.0,
                        current_substate=self._active_substate_name,
                    ),
                    interrupted=(old_state == MacroState.AS_EMERGENCY),
                )
                self._active_mission_behavior = None

        elif new_state == MacroState.AS_READY:
            self._assi_status = ASSIStatus.CONTINUOUS_YELLOW
            self._ready_start_time = now
            self._premature_res_detected = False
            self._service_brake_cmd = True
            self._ebs_trigger_cmd = False

        elif new_state == MacroState.AS_FINISHED:
            self._assi_status = ASSIStatus.FLASHING_BLUE
            self._ebs_trigger_cmd = True
            self._service_brake_cmd = False
            if self._active_mission_behavior is not None:
                self._active_mission_behavior.on_exit(
                    MissionContext(
                        telemetry=VcuTelemetryMessage(
                            timestamp=now,
                            ebs_status=EbsStatus.ACTIVATED,
                            asms_status=True,
                            dbs_check_ok=True,
                            ts_status=True,
                            ami_mission=Mission.NOT_SELECTED,
                            brake_status=True,
                            sdc_status=True,
                            res_go_signal=False,
                        ),
                        dt=0.0,
                        mission_time=now - self._mission_start_time,
                        current_substate=self._active_substate_name,
                    ),
                    interrupted=False,
                )

        elif new_state == MacroState.MANUAL_DRIVING:
            self._assi_status = ASSIStatus.OFF
            self._ebs_trigger_cmd = False
            self._service_brake_cmd = False

    def _transition_to_driving(self, telemetry: VcuTelemetryMessage, now: float) -> None:
        """Initialize driving state and instantiate Level 2 mission."""
        self._state = MacroState.AS_DRIVING
        self._assi_status = ASSIStatus.FLASHING_YELLOW
        self._state_entry_time = now
        self._mission_start_time = now
        self._service_brake_cmd = False
        self._ebs_trigger_cmd = False
        self._standstill_detector.reset()

        # Instantiate mission handler if available
        self._active_mission_behavior = MissionRegistry.create(telemetry.ami_mission)
        if self._active_mission_behavior is not None:
            setattr(self._active_mission_behavior, "_mission_enum", telemetry.ami_mission)
            self._active_substate_name = getattr(
                self._active_mission_behavior, "current_substate", "INIT"
            )
            if hasattr(self._active_substate_name, "value"):
                self._active_substate_name = self._active_substate_name.value
            context = MissionContext(
                telemetry=telemetry,
                dt=0.0,
                mission_time=0.0,
                current_substate=self._active_substate_name,
            )
            self._active_mission_behavior.on_enter(context)
        else:
            self._active_substate_name = "DRIVING_DEFAULT"

    def _transition_to_emergency(self, fault: DiagnosticErrorCode, now: float) -> None:
        """Latch into AS_Emergency with specific fault code."""
        self._state = MacroState.AS_EMERGENCY
        self._assi_status = ASSIStatus.CONTINUOUS_BLUE
        self._error_code |= int(fault)
        self._ebs_trigger_cmd = True
        self._service_brake_cmd = True
        self._state_entry_time = now

        if self._active_mission_behavior is not None:
            try:
                self._active_mission_behavior.on_exit(
                    MissionContext(
                        telemetry=VcuTelemetryMessage(
                            timestamp=now,
                            ebs_status=EbsStatus.ACTIVATED,
                            asms_status=False,
                            dbs_check_ok=False,
                            ts_status=False,
                            ami_mission=Mission.NOT_SELECTED,
                            brake_status=True,
                            sdc_status=False,
                            res_go_signal=False,
                        ),
                        dt=0.0,
                        mission_time=now - self._mission_start_time,
                        current_substate=self._active_substate_name,
                    ),
                    interrupted=True,
                )
            except Exception:
                pass
