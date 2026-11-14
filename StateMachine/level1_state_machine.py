"""Deterministic Level 1 Macro-State Machine for FSAE Driverless.

Every transition below maps to an arrow in flowchart.png unless marked
"(safety extension)". Keep it that way: when the diagram changes, change both.

    AS_Off       -> AS_Ready        EBS armed & ASMS on & DBS ok & TS on & mission & brakes engaged
    AS_Off       -> Manual_Driving  manual mission & EBS unavailable & ASMS off & TS on
    Manual       -> AS_Off          TS off
    AS_Ready     -> AS_Off          ASMS off & brakes released
    AS_Ready     -> AS_Driving      RES Go (fresh press) & 5 s passed
    AS_Ready     -> AS_Emergency    EBS activated (safety extension: also SDC open, VCU link lost)
    AS_Driving   -> AS_Finished     mission finished & velocity = 0
    AS_Driving   -> AS_Emergency    EBS activated (safety extension: SDC open, ASMS/TS drop,
                                    VCU or Level 2 link lost, Level 2 crash or e-stop,
                                    no standstill after mission finished)
    AS_Finished  -> AS_Off          ASMS off & brakes released
    AS_Emergency -> AS_Off          EBS sound off & ASMS off & brakes released
"""

import logging
from typing import Dict, Optional, Set, Tuple

import numpy as np

from Pipeline.messages import (
    MacroState,
    DSSIStatus,
    EbsStatus,
    Mission,
    DiagnosticErrorCode,
    VcuTelemetryMessage,
    ActuationCommandMessage,
    Level2StatusMessage,
    StateMachineStatusMessage,
)
from StateMachine.constants import (
    STATE_MACHINE_DEFAULT_DT_S,
    AS_READY_LOCKOUT_DURATION_S,
    VCU_WATCHDOG_TIMEOUT_S,
    LEVEL2_WATCHDOG_TIMEOUT_S,
    FINISH_STANDSTILL_TIMEOUT_S,
    HEARTBEAT_MAX,
)
from StateMachine.standstill_detector import StandstillDetector
from StateMachine.gatekeeper import gate_command
from StateMachine.level2.base import MissionBehavior, MissionContext, SubStateResult
from StateMachine.level2.registry import MissionRegistry

logger = logging.getLogger(__name__)

Outputs = Tuple[StateMachineStatusMessage, ActuationCommandMessage]

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

# States in which losing the VCU link is an emergency. AS_Finished is excluded on
# purpose: the car is stopped with EBS engaged and flowchart.png has no
# AS_Finished -> AS_Emergency arrow.
VCU_WATCHDOG_STATES = (MacroState.AS_READY, MacroState.AS_DRIVING)

# Substate label reported while an external planner (not a registered mission) drives.
EXTERNAL_PLANNER_SUBSTATE = "EXTERNAL_PLANNER"

_BIT_INDICES: np.ndarray = np.arange(16, dtype=np.int32)


def unpack_diagnostic_flags(error_code: int) -> np.ndarray:
    """Vectorized unpacking of an integer error bitmask into a 16-element bool array."""
    return ((int(error_code) >> _BIT_INDICES) & 1).astype(bool)


class Level1StateMachine:
    """FSAE Driverless Level 1 macro-state machine. See the module docstring for the arrows."""

    def __init__(self, mission_behavior: Optional[MissionBehavior] = None) -> None:
        self._state = MacroState.AS_OFF
        self._error_code = DiagnosticErrorCode.NONE
        self._heartbeat = 0

        # Mission committed on entering AS_Ready; it is what actually runs on RES Go.
        self._mission = Mission.NOT_SELECTED
        self._injected_behavior = mission_behavior
        self._behavior: Optional[MissionBehavior] = None
        self._substate = "NONE"

        self._ready_start_time = 0.0
        self._mission_start_time = 0.0
        self._finish_requested_time: Optional[float] = None
        self._premature_res_detected = False

        self._last_tick_time: Optional[float] = None
        self._last_telemetry: Optional[VcuTelemetryMessage] = None
        self._standstill = StandstillDetector()

    # -----------------------------------------------------------------------
    # Public read-only state
    # -----------------------------------------------------------------------

    @property
    def current_state(self) -> MacroState:
        return self._state

    @property
    def dssi_status(self) -> DSSIStatus:
        return STATE_DSSI_MAP[self._state]

    @property
    def error_code(self) -> DiagnosticErrorCode:
        return self._error_code

    @property
    def heartbeat(self) -> int:
        return self._heartbeat

    @property
    def active_mission(self) -> Mission:
        return self._mission

    @property
    def active_substate_name(self) -> str:
        return self._substate

    @property
    def ebs_trigger_cmd(self) -> bool:
        return self._state in (MacroState.AS_FINISHED, MacroState.AS_EMERGENCY)

    @property
    def service_brake_cmd(self) -> bool:
        return self._state in (MacroState.AS_READY, MacroState.AS_EMERGENCY)

    @property
    def ready_timer_elapsed(self) -> float:
        if self._state != MacroState.AS_READY or self._last_tick_time is None:
            return 0.0
        return max(0.0, self._last_tick_time - self._ready_start_time)

    # -----------------------------------------------------------------------
    # Entry points
    # -----------------------------------------------------------------------

    def step(
        self,
        telemetry: VcuTelemetryMessage,
        external_actuation: Optional[ActuationCommandMessage] = None,
        external_status: Optional[Level2StatusMessage] = None,
    ) -> Outputs:
        """Run one cycle on a fresh VCU telemetry packet.

        ``external_actuation`` / ``external_status`` come from an external Level 2
        planner and are only used when no registered mission is running.
        """
        now = telemetry.timestamp
        # Keep the time as a float: callers may reuse and mutate message objects.
        dt = now - self._last_tick_time if self._last_tick_time is not None else STATE_MACHINE_DEFAULT_DT_S
        self._last_tick_time = now
        self._last_telemetry = telemetry
        self._standstill.push(now, telemetry.velocity)

        # A late packet proves the link dropped for longer than allowed.
        if self._state in VCU_WATCHDOG_STATES and dt > VCU_WATCHDOG_TIMEOUT_S:
            self._trip_emergency(DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU, now)

        raw_cmd = None
        if self._state == MacroState.AS_OFF:
            self._step_as_off(telemetry, now)
        elif self._state == MacroState.AS_READY:
            self._step_as_ready(telemetry, now)
        elif self._state == MacroState.AS_DRIVING:
            raw_cmd = self._step_as_driving(telemetry, dt, now, external_actuation, external_status)
        elif self._state == MacroState.AS_FINISHED:
            self._step_as_finished(telemetry, now)
        elif self._state == MacroState.AS_EMERGENCY:
            self._step_as_emergency(telemetry, now)
        elif self._state == MacroState.MANUAL_DRIVING:
            self._step_manual_driving(telemetry, now)

        return self._outputs(now, raw_cmd)

    def handle_telemetry_timeout(self, now: float) -> Optional[Outputs]:
        """Called when no telemetry has arrived for VCU_WATCHDOG_TIMEOUT_S.

        Unlike the packet-gap check in step(), this fires while the link is still
        dead. Returns the outputs to publish, or None if this state ignores link loss.
        """
        if self._state not in VCU_WATCHDOG_STATES:
            return None
        self._trip_emergency(DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU, now)
        return self._outputs(now, None)

    def request_emergency_trip(self, fault: DiagnosticErrorCode, timestamp: float) -> None:
        """Trip immediately into AS_Emergency with a specific fault code."""
        self._trip_emergency(fault, timestamp)

    # -----------------------------------------------------------------------
    # Per-state logic
    # -----------------------------------------------------------------------

    def _step_as_off(self, tel: VcuTelemetryMessage, now: float) -> None:
        if (
            tel.ami_mission == Mission.MANUAL
            and tel.ebs_status == EbsStatus.UNAVAILABLE
            and not tel.dsms_status
            and tel.ts_status
        ):
            self._set_state(MacroState.MANUAL_DRIVING, now)
            return

        if (
            tel.ebs_status == EbsStatus.ARMED
            and tel.dsms_status
            and tel.dbs_check_ok
            and tel.ts_status
            and tel.ami_mission in VALID_DRIVERLESS_MISSIONS
            and tel.brake_status
            and tel.sdc_status
        ):
            self._mission = tel.ami_mission
            self._set_state(MacroState.AS_READY, now)

    def _step_as_ready(self, tel: VcuTelemetryMessage, now: float) -> None:
        if not tel.sdc_status:
            self._trip_emergency(DiagnosticErrorCode.SDC_OPEN, now)
            return
        if tel.ebs_status == EbsStatus.ACTIVATED:
            self._trip_emergency(DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED, now)
            return
        if not tel.dsms_status and not tel.brake_status:
            self._set_state(MacroState.AS_OFF, now)
            return

        # A press during the 5 s lockout must be released before a press counts.
        if now - self._ready_start_time < AS_READY_LOCKOUT_DURATION_S:
            self._premature_res_detected |= tel.res_go_signal
            return
        if self._premature_res_detected:
            self._premature_res_detected = tel.res_go_signal
            return

        if (
            tel.res_go_signal
            and tel.ebs_status == EbsStatus.ARMED
            and tel.dsms_status
            and tel.ts_status
            and tel.dbs_check_ok
        ):
            self._set_state(MacroState.AS_DRIVING, now)
            self._start_mission(tel, now)

    def _step_as_driving(
        self,
        tel: VcuTelemetryMessage,
        dt: float,
        now: float,
        external_cmd: Optional[ActuationCommandMessage],
        external_status: Optional[Level2StatusMessage],
    ) -> Optional[ActuationCommandMessage]:
        hardware_faults = (
            (not tel.sdc_status, DiagnosticErrorCode.SDC_OPEN),
            (tel.ebs_status == EbsStatus.ACTIVATED, DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED),
            (not tel.dsms_status, DiagnosticErrorCode.DSMS_UNEXPECTED_DROP),
            (not tel.ts_status, DiagnosticErrorCode.TS_INACTIVE),
        )
        for failed, fault in hardware_faults:
            if failed:
                self._trip_emergency(fault, now)
                return None

        if self._behavior is not None:
            result = self._run_mission(tel, dt, now)
        else:
            result = self._external_planner_result(now, external_cmd, external_status)
        if result is None:  # Level 2 failed and already tripped
            return None

        if result.next_substate:
            self._substate = result.next_substate
        if result.emergency_stop:
            self._trip_emergency(DiagnosticErrorCode.LEVEL2_EMERGENCY_STOP, now)
            return None

        if not result.mission_finished:
            self._finish_requested_time = None
            return result.actuation_command

        if self._finish_requested_time is None:
            self._finish_requested_time = now
        if self._standstill.is_standstill():
            self._set_state(MacroState.AS_FINISHED, now)
            return None
        if now - self._finish_requested_time > FINISH_STANDSTILL_TIMEOUT_S:
            self._trip_emergency(DiagnosticErrorCode.STANDSTILL_TIMEOUT, now)
            return None
        return result.actuation_command

    def _step_as_finished(self, tel: VcuTelemetryMessage, now: float) -> None:
        if not tel.dsms_status and not tel.brake_status:
            self._set_state(MacroState.AS_OFF, now)

    def _step_as_emergency(self, tel: VcuTelemetryMessage, now: float) -> None:
        if not tel.ebs_sound_active and not tel.dsms_status and not tel.brake_status:
            logger.info("EMERGENCY CLEARED at t=%.3fs (faults were %r)", now, self._error_code)
            self._set_state(MacroState.AS_OFF, now)

    def _step_manual_driving(self, tel: VcuTelemetryMessage, now: float) -> None:
        if not tel.ts_status:
            self._set_state(MacroState.AS_OFF, now)

    # -----------------------------------------------------------------------
    # Level 2 helpers
    # -----------------------------------------------------------------------

    def _start_mission(self, tel: VcuTelemetryMessage, now: float) -> None:
        self._mission_start_time = now
        self._finish_requested_time = None
        self._standstill.reset()
        if self._injected_behavior is not None:
            self._behavior = self._injected_behavior
        else:
            self._behavior = MissionRegistry.create(self._mission)
        if self._behavior is None:
            self._substate = EXTERNAL_PLANNER_SUBSTATE
            return

        self._substate = "INIT"
        try:
            self._behavior.on_enter(self._context(tel, dt=0.0, now=now))
        except Exception:
            logger.exception("Level 2 on_enter raised for mission %s", self._mission.name)
            self._trip_emergency(DiagnosticErrorCode.LEVEL2_EXCEPTION, now)

    def _run_mission(self, tel: VcuTelemetryMessage, dt: float, now: float) -> Optional[SubStateResult]:
        try:
            return self._behavior.update(self._context(tel, dt, now))
        except Exception:
            logger.exception("Level 2 update raised for mission %s", self._mission.name)
            self._trip_emergency(DiagnosticErrorCode.LEVEL2_EXCEPTION, now)
            return None

    def _external_planner_result(
        self,
        now: float,
        cmd: Optional[ActuationCommandMessage],
        status: Optional[Level2StatusMessage],
    ) -> Optional[SubStateResult]:
        if cmd is None or now - cmd.timestamp > LEVEL2_WATCHDOG_TIMEOUT_S:
            self._trip_emergency(DiagnosticErrorCode.WATCHDOG_TIMEOUT_LEVEL2, now)
            return None
        # Ignore status left over from a previous run.
        if status is None or status.timestamp < self._mission_start_time:
            return SubStateResult(actuation_command=cmd)
        return SubStateResult(
            next_substate=status.active_substate,
            mission_finished=status.mission_finished,
            emergency_stop=status.emergency_stop,
            actuation_command=cmd,
        )

    def _context(self, tel: VcuTelemetryMessage, dt: float, now: float) -> MissionContext:
        return MissionContext(
            telemetry=tel,
            dt=dt,
            mission_time=now - self._mission_start_time,
            current_substate=self._substate,
        )

    # -----------------------------------------------------------------------
    # Transitions and outputs
    # -----------------------------------------------------------------------

    def _set_state(self, new_state: MacroState, now: float) -> None:
        old_state = self._state
        if old_state == new_state:
            return

        logger.info(
            "STATE TRANSITION: %s -> %s [t=%.3fs, mission=%s, substate=%s]",
            old_state.name,
            new_state.name,
            now,
            self._mission.name,
            self._substate,
        )
        self._state = new_state

        if old_state == MacroState.AS_DRIVING:
            self._stop_mission(now, interrupted=(new_state == MacroState.AS_EMERGENCY))
        if new_state == MacroState.AS_READY:
            self._ready_start_time = now
            self._premature_res_detected = False
        elif new_state == MacroState.AS_OFF:
            self._error_code = DiagnosticErrorCode.NONE
            self._mission = Mission.NOT_SELECTED
            self._substate = "NONE"

    def _stop_mission(self, now: float, interrupted: bool) -> None:
        if self._behavior is not None and self._last_telemetry is not None:
            try:
                self._behavior.on_exit(self._context(self._last_telemetry, 0.0, now), interrupted=interrupted)
            except Exception:
                logger.exception("Level 2 on_exit raised for mission %s", self._mission.name)
        self._behavior = None

    def _trip_emergency(self, fault: DiagnosticErrorCode, now: float) -> None:
        self._error_code |= fault
        logger.critical(
            "EMERGENCY TRIP: %s [faults=%r, from=%s, mission=%s, substate=%s, t=%.3fs]",
            fault.name,
            self._error_code,
            self._state.name,
            self._mission.name,
            self._substate,
            now,
        )
        self._set_state(MacroState.AS_EMERGENCY, now)

    def _outputs(self, now: float, raw_cmd: Optional[ActuationCommandMessage]) -> Outputs:
        self._heartbeat = (self._heartbeat + 1) & HEARTBEAT_MAX
        status = StateMachineStatusMessage(
            timestamp=now,
            current_state=self._state,
            dssi_status=self.dssi_status,
            error_code=int(self._error_code),
            error_flags=unpack_diagnostic_flags(self._error_code),
            heartbeat=self._heartbeat,
            ebs_trigger_cmd=self.ebs_trigger_cmd,
            service_brake_cmd=self.service_brake_cmd,
            active_mission=self._mission,
            active_substate=self._substate,
            ready_timer_elapsed=self.ready_timer_elapsed,
        )
        return status, gate_command(self._state, raw_cmd, now)
