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
    """Unpacks an error bitmask into a boolean array, one element per bit.

    Args:
        error_code: DiagnosticErrorCode bitmask.

    Returns:
        A 16-element bool array where index i is True if bit i is set.
    """
    return ((int(error_code) >> _BIT_INDICES) & 1).astype(bool)


class Level1StateMachine:
    """Deterministic FSAE Driverless Level 1 macro-state machine.

    Owns the rule-mandated safety states and gates every Level 3 actuation
    command. The module docstring lists each transition and whether it comes
    from flowchart.png or is a safety extension.

    Typical usage example:

        state_machine = Level1StateMachine()
        status, actuation = state_machine.step(telemetry=tel)

    Args:
        mission_behavior: Optional Level 2 mission to run in every AS_DRIVING
            run. If None, the mission is created from the MissionRegistry for
            the mission committed on entering AS_READY; if none is
            registered, an external planner is expected to drive.

    Attributes:
        current_state: The active MacroState.
        dssi_status: DSSI light state for the active macro-state.
        error_code: DiagnosticErrorCode bitmask of latched faults. Cleared on
            entering AS_OFF.
        heartbeat: Counter incremented on every output, wrapping at
            HEARTBEAT_MAX.
        active_mission: Mission committed on entering AS_READY, or
            NOT_SELECTED in AS_OFF.
        active_substate_name: Level 2 sub-state label, "EXTERNAL_PLANNER"
            when an external planner drives, or "NONE".
        ebs_trigger_cmd: True when EBS must be engaged (AS_FINISHED,
            AS_EMERGENCY).
        service_brake_cmd: True when the service brake must be held
            (AS_READY, AS_EMERGENCY).
        ready_timer_elapsed: Seconds spent in AS_READY, or 0.0 elsewhere.
    """

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
        """Runs one state machine cycle on a fresh VCU telemetry packet.

        Order of work: VCU packet-gap watchdog, the active state's checks and
        transitions, then actuation gating for the resulting state.

        Args:
            telemetry: Latest VCU telemetry packet.
            external_actuation: Latest command from an external Level 2
                planner. Only used in AS_DRIVING when no registered mission
                is running, and only if newer than LEVEL2_WATCHDOG_TIMEOUT_S.
            external_status: Latest status from that external planner, used
                to report its sub-state, mission finished or e-stop.

        Returns:
            A (StateMachineStatusMessage, ActuationCommandMessage) pair. The
            actuation command is always safe to send to the VCU.
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
        match self._state:
            case MacroState.AS_OFF:
                self._step_as_off(telemetry, now)
            case MacroState.AS_READY:
                self._step_as_ready(telemetry, now)
            case MacroState.AS_DRIVING:
                raw_cmd = self._step_as_driving(telemetry, dt, now, external_actuation, external_status)
            case MacroState.AS_FINISHED:
                self._step_as_finished(telemetry, now)
            case MacroState.AS_EMERGENCY:
                self._step_as_emergency(telemetry, now)
            case MacroState.MANUAL_DRIVING:
                self._step_manual_driving(telemetry, now)

        return self._outputs(now, raw_cmd)

    def handle_telemetry_timeout(self, now: float) -> Optional[Outputs]:
        """Trips the VCU watchdog when telemetry has stopped arriving.

        Called by StateMachineNode's watchdog thread after
        VCU_WATCHDOG_TIMEOUT_S without a packet. Unlike the packet-gap check
        in step(), this fires while the link is still dead.

        Args:
            now: Trip time on the telemetry clock.

        Returns:
            The (status, actuation) outputs to publish, or None if the current
            state does not watch the VCU link (only AS_READY and AS_DRIVING do).
        """
        if self._state not in VCU_WATCHDOG_STATES:
            return None
        self._trip_emergency(DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU, now)
        return self._outputs(now, None)

    def request_emergency_trip(self, fault: DiagnosticErrorCode, timestamp: float) -> None:
        """Trips immediately into AS_EMERGENCY.

        Args:
            fault: Fault flag to latch into error_code.
            timestamp: Trip time on the telemetry clock.
        """
        self._trip_emergency(fault, timestamp)

    # -----------------------------------------------------------------------
    # Per-state logic
    # -----------------------------------------------------------------------

    def _step_as_off(self, tel: VcuTelemetryMessage, now: float) -> None:
        """Enters MANUAL_DRIVING or AS_READY when the flowchart conditions hold.

        Manual driving is checked first. Entering AS_READY commits the
        selected mission for the rest of the run.

        Args:
            tel: Latest VCU telemetry packet.
            now: Current telemetry timestamp.
        """
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
        """Evaluates AS_READY faults, exit, and the RES Go start sequence.

        Checks run in priority order:

            1. Faults: an open SDC or an activated EBS trips AS_EMERGENCY.
            2. Clean exit: ASMS off with brakes released returns to AS_OFF.
            3. Lockout: RES Go is ignored for AS_READY_LOCKOUT_DURATION_S, and
               a press during the lockout is remembered as premature.
            4. Premature press: it must be released before a new press counts,
               so holding RES Go through the lockout cannot start the mission.
            5. Start: a fresh press enters AS_DRIVING if EBS is armed and ASMS,
               TS and the DBS check are still good.

        Args:
            tel: Latest VCU telemetry packet.
            now: Current telemetry timestamp.
        """
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
        """Runs Level 2 and evaluates AS_DRIVING faults and completion.

        Hardware faults trip first. Then the registered mission (or the
        external planner) produces a result: an e-stop trips AS_EMERGENCY, and
        "mission finished" enters AS_FINISHED once the car is at standstill,
        or trips STANDSTILL_TIMEOUT if it is still moving after
        FINISH_STANDSTILL_TIMEOUT_S.

        Args:
            tel: Latest VCU telemetry packet.
            dt: Seconds since the previous packet.
            now: Current telemetry timestamp.
            external_cmd: Latest external planner command, if any.
            external_status: Latest external planner status, if any.

        Returns:
            The requested actuation command, or None to stop the car.
        """
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
        """Returns to AS_OFF once ASMS is off and the brakes are released.

        Args:
            tel: Latest VCU telemetry packet.
            now: Current telemetry timestamp.
        """
        if not tel.dsms_status and not tel.brake_status:
            self._set_state(MacroState.AS_OFF, now)

    def _step_as_emergency(self, tel: VcuTelemetryMessage, now: float) -> None:
        """Returns to AS_OFF once EBS sound, ASMS and brakes are all off.

        Args:
            tel: Latest VCU telemetry packet.
            now: Current telemetry timestamp.
        """
        if not tel.ebs_sound_active and not tel.dsms_status and not tel.brake_status:
            logger.info("EMERGENCY CLEARED at t=%.3fs (faults were %r)", now, self._error_code)
            self._set_state(MacroState.AS_OFF, now)

    def _step_manual_driving(self, tel: VcuTelemetryMessage, now: float) -> None:
        """Returns to AS_OFF when the tractive system turns off.

        Args:
            tel: Latest VCU telemetry packet.
            now: Current telemetry timestamp.
        """
        if not tel.ts_status:
            self._set_state(MacroState.AS_OFF, now)

    # -----------------------------------------------------------------------
    # Level 2 helpers
    # -----------------------------------------------------------------------

    def _start_mission(self, tel: VcuTelemetryMessage, now: float) -> None:
        """Creates and enters the Level 2 mission right after RES Go.

        Uses the injected mission if there is one, otherwise the registry.
        With neither, an external planner is expected to drive. A crash in
        on_enter trips LEVEL2_EXCEPTION.

        Args:
            tel: Telemetry packet that triggered AS_DRIVING.
            now: Current telemetry timestamp.
        """
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
        """Calls the registered mission's update, tripping on any exception.

        Args:
            tel: Latest VCU telemetry packet.
            dt: Seconds since the previous packet.
            now: Current telemetry timestamp.

        Returns:
            The mission's SubStateResult, or None if it raised.
        """
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
        """Builds a SubStateResult from an external planner's messages.

        Args:
            now: Current telemetry timestamp.
            cmd: Latest planner command. Missing or older than
                LEVEL2_WATCHDOG_TIMEOUT_S trips WATCHDOG_TIMEOUT_LEVEL2.
            status: Latest planner status. Ignored if it predates this run.

        Returns:
            The planner's result, or None if the Level 2 watchdog tripped.
        """
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
        """Builds the MissionContext passed to Level 2 callbacks."""
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
        """Performs a transition and its entry and exit side effects.

        Leaving AS_DRIVING calls the mission's on_exit. Entering AS_READY
        starts the lockout timer. Entering AS_OFF clears faults and the
        committed mission. A transition to the current state does nothing.

        Args:
            new_state: State to enter.
            now: Current telemetry timestamp.
        """
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
        """Calls the mission's on_exit and drops it. Exceptions are logged.

        Args:
            now: Current telemetry timestamp.
            interrupted: True when AS_DRIVING ends in AS_EMERGENCY.
        """
        if self._behavior is not None and self._last_telemetry is not None:
            try:
                self._behavior.on_exit(self._context(self._last_telemetry, 0.0, now), interrupted=interrupted)
            except Exception:
                logger.exception("Level 2 on_exit raised for mission %s", self._mission.name)
        self._behavior = None

    def _trip_emergency(self, fault: DiagnosticErrorCode, now: float) -> None:
        """Latches a fault, logs it with the running mission, and enters AS_EMERGENCY.

        Args:
            fault: Fault flag to latch into error_code.
            now: Current telemetry timestamp.
        """
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
        """Builds the status message and the gated actuation command.

        Args:
            now: Current telemetry timestamp.
            raw_cmd: Requested actuation, or None to stop the car.

        Returns:
            A (StateMachineStatusMessage, ActuationCommandMessage) pair.
        """
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
