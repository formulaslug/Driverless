"""Deterministic Level 1 Macro-State Machine for FSAE Driverless."""

import logging
from typing import Optional, Tuple, Set, Dict
import numpy as np

import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PIPELINE_DIR = os.path.join(ROOT_DIR, 'Pipeline')
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from messages import (
    MacroState,
    DSSIStatus,
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

logger = logging.getLogger(__name__)


VALID_DRIVERLESS_MISSIONS: Set[Mission] = {
    Mission.INSPECTION,
    Mission.ACCELERATION,
    Mission.SKIDPAD,
    Mission.AUTOCROSS,
    Mission.TRACKDRIVE,
}

STATE_DSSI_MAP: Dict[MacroState, DSSIStatus] = {
    MacroState.AS_OFF: DSSIStatus.OFF,
    MacroState.AS_READY: DSSIStatus.CONTINUOUS_YELLOW,
    MacroState.AS_DRIVING: DSSIStatus.FLASHING_YELLOW,
    MacroState.AS_FINISHED: DSSIStatus.FLASHING_BLUE,
    MacroState.AS_EMERGENCY: DSSIStatus.CONTINUOUS_BLUE,
    MacroState.MANUAL_DRIVING: DSSIStatus.OFF,
}

_NUM_DIAGNOSTIC_BITS: int = 16
_BIT_INDICES: np.ndarray = np.arange(_NUM_DIAGNOSTIC_BITS, dtype=np.int32)


def unpack_diagnostic_flags(error_code: int) -> np.ndarray:
    """Vectorized unpacking of integer error bitmask into a NumPy boolean array."""
    return ((error_code >> _BIT_INDICES) & 1).astype(bool)


class Level1StateMachine:
    """FSAE Driverless Level 1 Macro-State Machine.
    
    Governs mandatory competition safety states:
    - AS_Off: System powered off / safe neutral.
    - AS_Ready: Preconditions verified, 5.0s warning timer, service brake clamped.
    - AS_Driving: Autonomous mission active, 50 Hz watchdogs, Level 3 gated.
    - AS_Finished: Mission complete, verified standstill, EBS locked.
    - AS_Emergency: Safety circuit open or critical fault; latched until manual reset.
    - Manual_Driving: TS active with manual mission selected and DSMS off.
    """

    def __init__(self, mission_behavior: Optional[MissionBehavior] = None) -> None:
        self._state: MacroState = MacroState.AS_OFF
        self._dssi_status: DSSIStatus = DSSIStatus.OFF
        self._error_code: int = DiagnosticErrorCode.NONE

        # Timers and telemetry tracking
        self._state_entry_time: float = 0.0
        self._ready_start_time: float = 0.0
        self._mission_start_time: float = 0.0
        self._last_vcu_telemetry_time: Optional[float] = None
        self._last_tick_time: Optional[float] = None

        # Lockout and button flags
        self._premature_res_detected: bool = False

        # Level 2 mission runner (supports optional dependency injection)
        self._injected_mission_behavior: Optional[MissionBehavior] = mission_behavior
        self._active_mission_behavior: Optional[MissionBehavior] = mission_behavior
        self._active_substate_name: str = "NONE"

        # Hardware command outputs
        self._heartbeat: int = 0
        self._ebs_trigger_cmd: bool = False
        self._service_brake_cmd: bool = False

        # Standstill detection buffer
        self._standstill_detector: StandstillDetector = StandstillDetector()

    # -----------------------------------------------------------------------
    # Public Properties
    # -----------------------------------------------------------------------

    @property
    def current_state(self) -> MacroState:
        return self._state

    @property
    def dssi_status(self) -> DSSIStatus:
        return self._dssi_status

    @property
    def error_code(self) -> int:
        return self._error_code

    @property
    def heartbeat(self) -> int:
        return self._heartbeat

    @property
    def ready_timer_elapsed(self) -> float:
        if self._state == MacroState.AS_READY:
            return max(0.0, (self._last_tick_time or 0.0) - self._ready_start_time)
        return 0.0

    @property
    def active_mission(self) -> Optional[Mission]:
        if self._active_mission_behavior is not None:
            return getattr(self._active_mission_behavior, "_mission_enum", None)
        return None

    @property
    def active_substate_name(self) -> str:
        return self._active_substate_name

    # -----------------------------------------------------------------------
    # External Fault & Reset Controls
    # -----------------------------------------------------------------------

    def request_emergency_trip(self, fault: DiagnosticErrorCode, timestamp: float) -> None:
        """Trip immediately into AS_Emergency with specific fault code."""
        self._trip_emergency(fault, timestamp)

    def request_reset(self, telemetry: VcuTelemetryMessage) -> bool:
        """Attempt to clear AS_Emergency latch and return to AS_Off.
        
        Per FSAE rules and flowchart.png:
        Transition requires: EBS Sound Off & ASMS (DSMS) Off & Brakes Released.
        Note: The vehicle horn / acoustic warning buzzer is managed directly by
        VCU hardware firmware and silences after the mandated countdown. The CAN
        bridge enforces this reset safety invariant via ebs_status != ACTIVATED,
        dsms_status == False, brake_status == False, and sdc_status == True.
        """
        if self._state != MacroState.AS_EMERGENCY:
            return False

        can_reset = (
            telemetry.ebs_status != EbsStatus.ACTIVATED
            and not telemetry.dsms_status
            and not telemetry.brake_status
            and telemetry.sdc_status
        )

        if can_reset:
            self._error_code = DiagnosticErrorCode.NONE
            self._set_state(MacroState.AS_OFF, telemetry, telemetry.timestamp)
            return True

        return False

    # -----------------------------------------------------------------------
    # Main Step Cycle
    # -----------------------------------------------------------------------

    def step(
        self,
        telemetry: VcuTelemetryMessage,
        external_actuation: Optional[ActuationCommandMessage] = None,
    ) -> Tuple[StateMachineStatusMessage, ActuationCommandMessage]:
        """Execute one 50 Hz state machine cycle."""
        now = telemetry.timestamp
        dt = (now - self._last_tick_time) if self._last_tick_time is not None else 0.02
        self._last_tick_time = now
        self._heartbeat = (self._heartbeat + 1) & HEARTBEAT_MAX

        # Update standstill buffer
        self._standstill_detector.push(now, telemetry.velocity)

        # 1. Comms Watchdog Check (active during AS_Ready, AS_Driving, AS_Finished)
        if self._state in (MacroState.AS_READY, MacroState.AS_DRIVING, MacroState.AS_FINISHED):
            if self._last_vcu_telemetry_time is not None:
                if (now - self._last_vcu_telemetry_time) > VCU_WATCHDOG_TIMEOUT_S:
                    self._trip_emergency(DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU, now)
        self._last_vcu_telemetry_time = now

        # 2. Evaluate State-Specific Logic
        raw_cmd = external_actuation

        if self._state == MacroState.AS_OFF:
            raw_cmd = self._step_as_off(telemetry, now)
        elif self._state == MacroState.AS_READY:
            raw_cmd = self._step_as_ready(telemetry, now)
        elif self._state == MacroState.AS_DRIVING:
            raw_cmd = self._step_as_driving(telemetry, dt, now, external_actuation)
        elif self._state == MacroState.AS_FINISHED:
            raw_cmd = self._step_as_finished(telemetry, now)
        elif self._state == MacroState.AS_EMERGENCY:
            raw_cmd = self._step_as_emergency(telemetry, now)
        elif self._state == MacroState.MANUAL_DRIVING:
            raw_cmd = self._step_manual_driving(telemetry, now)

        # 3. Gated Actuation Filter
        gated_cmd = ActuationGatekeeper.gate_command(
            state=self._state,
            requested_command=raw_cmd,
            timestamp=now,
        )

        status_msg = StateMachineStatusMessage(
            timestamp=now,
            current_state=self._state,
            dssi_status=self._dssi_status,
            error_code=self._error_code,
            error_flags=unpack_diagnostic_flags(self._error_code),
            heartbeat=self._heartbeat,
            ebs_trigger_cmd=self._ebs_trigger_cmd,
            service_brake_cmd=self._service_brake_cmd,
            active_mission=telemetry.ami_mission,
            active_substate=self._active_substate_name,
            ready_timer_elapsed=self.ready_timer_elapsed,
        )

        return status_msg, gated_cmd

    # -----------------------------------------------------------------------
    # State Logic Handlers
    # -----------------------------------------------------------------------

    def _step_as_off(self, tel: VcuTelemetryMessage, now: float) -> Optional[ActuationCommandMessage]:
        # Check Manual Driving transition
        if tel.ami_mission == Mission.MANUAL and tel.ebs_status == EbsStatus.UNAVAILABLE and not tel.dsms_status and tel.ts_status:
            self._set_state(MacroState.MANUAL_DRIVING, tel, now)
            return None

        # Check AS_Ready preconditions
        all_ready = (
            tel.ebs_status == EbsStatus.ARMED
            and tel.dsms_status
            and tel.dbs_check_ok
            and tel.ts_status
            and (tel.ami_mission in VALID_DRIVERLESS_MISSIONS)
            and tel.brake_status
            and tel.sdc_status
        )
        if all_ready:
            self._set_state(MacroState.AS_READY, tel, now)

        return None

    def _step_as_ready(self, tel: VcuTelemetryMessage, now: float) -> Optional[ActuationCommandMessage]:
        # Critical safety trips
        if not tel.sdc_status:
            self._trip_emergency(DiagnosticErrorCode.SDC_OPEN, now)
            return None
        if tel.ebs_status == EbsStatus.ACTIVATED:
            self._trip_emergency(DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED, now)
            return None

        # Clean exit: DSMS off & brakes released
        if (not tel.dsms_status) and (not tel.brake_status):
            self._set_state(MacroState.AS_OFF, tel, now)
            return None

        elapsed = now - self._ready_start_time

        # Enforce 5.0s lockout: any press before 5.0s is rejected
        if elapsed < AS_READY_LOCKOUT_DURATION_S:
            if tel.res_go_signal:
                self._premature_res_detected = True
            return None

        # At t >= 5.0s: premature press must be released first
        if self._premature_res_detected:
            if not tel.res_go_signal:
                self._premature_res_detected = False
            return None

        # Valid fresh RES Go press -> AS_Driving
        if tel.res_go_signal:
            preconditions_hold = (
                tel.ebs_status == EbsStatus.ARMED
                and tel.dsms_status
                and tel.ts_status
                and tel.sdc_status
                and tel.dbs_check_ok
            )
            if preconditions_hold:
                self._set_state(MacroState.AS_DRIVING, tel, now)

        return None

    def _step_as_driving(
        self, tel: VcuTelemetryMessage, dt: float, now: float, external_cmd: Optional[ActuationCommandMessage]
    ) -> Optional[ActuationCommandMessage]:
        # Hardware fault checks
        if not tel.sdc_status:
            self._trip_emergency(DiagnosticErrorCode.SDC_OPEN, now)
            return None
        if tel.ebs_status == EbsStatus.ACTIVATED:
            self._trip_emergency(DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED, now)
            return None
        if not tel.dsms_status:
            self._trip_emergency(DiagnosticErrorCode.DSMS_UNEXPECTED_DROP, now)
            return None
        if not tel.ts_status:
            self._trip_emergency(DiagnosticErrorCode.TS_INACTIVE, now)
            return None

        mission_time = now - self._mission_start_time
        context = MissionContext(
            telemetry=tel,
            dt=dt,
            mission_time=mission_time,
            current_substate=self._active_substate_name,
        )

        sub_result: Optional[SubStateResult] = None
        cmd_out = external_cmd

        if self._active_mission_behavior is not None:
            try:
                sub_result = self._active_mission_behavior.update(context)
                if sub_result.next_substate is not None:
                    self._active_substate_name = sub_result.next_substate
                if sub_result.actuation_command is not None:
                    cmd_out = sub_result.actuation_command
            except Exception as exc:
                logger.exception("Unhandled exception in Level 2 mission behavior update: %s", exc)
                self._trip_emergency(DiagnosticErrorCode.LEVEL2_EXCEPTION, now)
                return None

        # Level 2 emergency stop request
        if sub_result is not None and sub_result.emergency_stop:
            self._trip_emergency(DiagnosticErrorCode.LEVEL2_EXCEPTION, now)
            return None

        # Check mission completion & verified standstill -> AS_Finished
        if sub_result is not None and sub_result.mission_finished:
            if self._standstill_detector.is_standstill(
                required_duration_s=STANDSTILL_MIN_DURATION_S,
                threshold_mps=STANDSTILL_VELOCITY_THRESHOLD_MPS,
            ):
                self._set_state(MacroState.AS_FINISHED, tel, now)
                return None

        return cmd_out

    def _step_as_finished(self, tel: VcuTelemetryMessage, now: float) -> Optional[ActuationCommandMessage]:
        if (not tel.dsms_status) and (not tel.brake_status):
            self._set_state(MacroState.AS_OFF, tel, now)
        return None

    def _step_as_emergency(self, tel: VcuTelemetryMessage, now: float) -> Optional[ActuationCommandMessage]:
        return None

    def _step_manual_driving(self, tel: VcuTelemetryMessage, now: float) -> Optional[ActuationCommandMessage]:
        if not tel.ts_status:
            self._set_state(MacroState.AS_OFF, tel, now)
        return None

    # -----------------------------------------------------------------------
    # Transition Logic
    # -----------------------------------------------------------------------

    def _set_state(self, new_state: MacroState, tel: VcuTelemetryMessage, now: float) -> None:
        """Centralized state transition handler."""
        old_state = self._state
        self._state = new_state
        self._state_entry_time = now
        self._dssi_status = STATE_DSSI_MAP[new_state]

        # Clean up any active mission if leaving AS_Driving
        if old_state == MacroState.AS_DRIVING and new_state != MacroState.AS_DRIVING:
            if self._active_mission_behavior is not None:
                ctx = MissionContext(
                    telemetry=tel,
                    dt=0.0,
                    mission_time=now - self._mission_start_time,
                    current_substate=self._active_substate_name,
                )
                try:
                    self._active_mission_behavior.on_exit(ctx, interrupted=(new_state == MacroState.AS_EMERGENCY))
                except Exception as exc:
                    logger.exception("Error executing mission on_exit callback: %s", exc)
                if new_state == MacroState.AS_OFF:
                    self._active_mission_behavior = self._injected_mission_behavior

        # State entry side-effects
        if new_state == MacroState.AS_OFF:
            self._ebs_trigger_cmd = False
            self._service_brake_cmd = False
            self._active_substate_name = "NONE"
            self._premature_res_detected = False

        elif new_state == MacroState.AS_READY:
            self._ready_start_time = now
            self._premature_res_detected = False
            self._service_brake_cmd = True
            self._ebs_trigger_cmd = False

        elif new_state == MacroState.AS_DRIVING:
            self._mission_start_time = now
            self._service_brake_cmd = False
            self._ebs_trigger_cmd = False
            self._standstill_detector.reset()

            if self._injected_mission_behavior is not None:
                self._active_mission_behavior = self._injected_mission_behavior
            else:
                self._active_mission_behavior = MissionRegistry.create(tel.ami_mission)

            if self._active_mission_behavior is not None:
                setattr(self._active_mission_behavior, "_mission_enum", tel.ami_mission)
                sub_name = getattr(self._active_mission_behavior, "current_substate", "INIT")
                self._active_substate_name = sub_name.value if hasattr(sub_name, "value") else str(sub_name)
                ctx = MissionContext(telemetry=tel, dt=0.0, mission_time=0.0, current_substate=self._active_substate_name)
                self._active_mission_behavior.on_enter(ctx)
            else:
                self._active_substate_name = "DRIVING_DEFAULT"

        elif new_state == MacroState.AS_FINISHED:
            self._ebs_trigger_cmd = True
            self._service_brake_cmd = False

        elif new_state == MacroState.AS_EMERGENCY:
            self._ebs_trigger_cmd = True
            self._service_brake_cmd = True

        elif new_state == MacroState.MANUAL_DRIVING:
            self._ebs_trigger_cmd = False
            self._service_brake_cmd = False

    def _trip_emergency(self, fault: DiagnosticErrorCode, now: float) -> None:
        """Latch into AS_Emergency immediately with fault bitmask."""
        self._error_code |= int(fault)
        dummy_tel = VcuTelemetryMessage(timestamp=now, ebs_status=EbsStatus.ACTIVATED, dsms_status=False)
        self._set_state(MacroState.AS_EMERGENCY, dummy_tel, now)
