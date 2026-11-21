"""Deterministic Level 1 Macro-State Machine for FSAE Driverless.

Every transition below maps to an arrow in flowchart.png unless marked
"(safety extension)". Keep it that way: when the diagram changes, change both.

    DS_Off       -> DS_Ready        EBS armed & DSMS on & DBS ok & TS on & mission & brakes engaged
    DS_Off       -> Manual_Driving  manual mission & EBS unavailable & DSMS off & TS on
    Manual       -> DS_Off          TS off
    DS_Ready     -> DS_Off          DSMS off & brakes released
    DS_Ready     -> DS_Driving      RES Go (fresh press) & 5s passed
    DS_Ready     -> DS_Emergency    EBS activated (safety extension: also SDC open, VCU link lost)
    DS_Driving   -> DS_Finished     mission finished & velocity = 0
    DS_Driving   -> DS_Emergency    EBS activated (safety extension: SDC open at RSS, DSMS/TS drop,
                                    VCU or Level 2 link lost, Level 2 crash or e-stop,
                                    no standstill after mission finished)
    DS_Finished  -> DS_Off          DSMS off & brakes released
    DS_Emergency -> DS_Off          EBS sound off & DSMS off & brakes released
"""

import logging
from typing import Dict, Optional, Set, Tuple, Union

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
    DS_READY_LOCKOUT_DURATION_S,
    VCU_WATCHDOG_TIMEOUT_S,
    LEVEL2_WATCHDOG_TIMEOUT_S,
    FINISH_STANDSTILL_TIMEOUT_S,
    HEARTBEAT_MAX,
)
from StateMachine.standstill_detector import StandstillDetector
from StateMachine.gatekeeper import gate_command
from StateMachine.level2.base import SubStateResult
from StateMachine.level2.planner_link import PlannerLink

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
    MacroState.DS_OFF:         DSSIStatus.OFF,
    MacroState.DS_READY:       DSSIStatus.CONTINUOUS_YELLOW,
    MacroState.DS_DRIVING:     DSSIStatus.FLASHING_YELLOW,
    MacroState.DS_FINISHED:    DSSIStatus.FLASHING_BLUE,
    MacroState.DS_EMERGENCY:   DSSIStatus.CONTINUOUS_BLUE,
    MacroState.MANUAL_DRIVING: DSSIStatus.OFF,
}

# States in which losing the VCU link is an emergency. DS_Finished is excluded on
# purpose: the car is stopped with EBS engaged and flowchart.png has no
# DS_Finished -> DS_Emergency arrow.
VCU_WATCHDOG_STATES = (MacroState.DS_READY, MacroState.DS_DRIVING)

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

        state_machine = Level1StateMachine(planner_link)
        status, actuation = state_machine.step(telemetry=tel)

    Args:
        planner_link: The planner link that will process ActuationCommandMessages
            and Level2StatusMessages.

    Attributes:
        current_state: The active MacroState.
        dssi_status: DSSI light state for the active macro-state.
        error_code: DiagnosticErrorCode bitmask of latched faults. Cleared on
            entering DS_OFF.
        heartbeat: Counter incremented on every output, wrapping at
            HEARTBEAT_MAX.
        active_mission: Mission committed on entering DS_READY, or
            NOT_SELECTED in DS_OFF.
        active_substate_name: Level 2 sub-state label.
        ebs_trigger_cmd: True when EBS must be engaged (DS_FINISHED,
            DS_EMERGENCY).
        service_brake_cmd: True when the service brake must be held
            (DS_READY, DS_EMERGENCY).
        ready_timer_elapsed: Seconds spent in DS_READY, or 0.0 elsewhere.
    """

    def __init__(self, planner_link: PlannerLink) -> None:
        self._state = MacroState.DS_OFF
        self._error_code = DiagnosticErrorCode.NONE
        self._heartbeat = 0

        # Mission committed on entering DS_Ready; it is what actually runs on RES Go.
        self._mission = Mission.NOT_SELECTED
        self._substate = "NONE"

        self._ready_start_time = 0.0
        self._finish_requested_time: Optional[float] = None
        self._premature_res_detected = False
        # False from a trip until an output reporting DS_EMERGENCY has been built.
        self._emergency_reported = True

        self._last_tick_time: Optional[float] = None
        self._last_telemetry: Optional[VcuTelemetryMessage] = None
        self._standstill = StandstillDetector()
        self._planner_link = planner_link

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
        return self._state in (MacroState.DS_FINISHED, MacroState.DS_EMERGENCY)

    @property
    def service_brake_cmd(self) -> bool:
        return self._state in (MacroState.DS_READY, MacroState.DS_EMERGENCY)

    @property
    def ready_timer_elapsed(self) -> float:
        if self._state != MacroState.DS_READY or self._last_tick_time is None:
            return 0.0
        return max(0.0, self._last_tick_time - self._ready_start_time)

    # -----------------------------------------------------------------------
    # Entry points
    # -----------------------------------------------------------------------

    def step(self, telemetry: VcuTelemetryMessage) -> Outputs:
        """Runs one state machine cycle on a fresh VCU telemetry packet.

        Order of work: VCU packet-gap watchdog, the active state's checks and
        transitions, then actuation gating for the resulting state.

        Args:
            telemetry: Latest VCU telemetry packet.

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
            case MacroState.DS_OFF:
                self._step_ds_off(telemetry, now)
            case MacroState.DS_READY:
                self._step_ds_ready(telemetry, now)
            case MacroState.DS_DRIVING:
                raw_cmd = self._step_ds_driving(telemetry, dt, now)
            case MacroState.DS_FINISHED:
                self._step_ds_finished(telemetry, now)
            case MacroState.DS_EMERGENCY:
                self._step_ds_emergency(telemetry, now)
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
            state does not watch the VCU link (only DS_READY and DS_DRIVING do).
        """
        if self._state not in VCU_WATCHDOG_STATES:
            return None
        self._trip_emergency(DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU, now)
        return self._outputs(now, None)

    def request_emergency_trip(self, fault: DiagnosticErrorCode, timestamp: float) -> None:
        """Trips immediately into DS_EMERGENCY.

        Args:
            fault: Fault flag to latch into error_code.
            timestamp: Trip time on the telemetry clock.
        """
        self._trip_emergency(fault, timestamp)

    # -----------------------------------------------------------------------
    # Per-state logic
    # -----------------------------------------------------------------------

    def _step_ds_off(self, tel: VcuTelemetryMessage, now: float) -> None:
        """Enters MANUAL_DRIVING or DS_READY when the flowchart conditions hold.

        Manual driving is checked first. Entering DS_READY commits the
        selected mission for the rest of the run.

        Args:
            tel: Latest VCU telemetry packet.
            now: Current telemetry timestamp.
        """
        if (
            tel.dmi_mission == Mission.MANUAL
            and tel.ebs_status == EbsStatus.UNAVAILABLE
            and not tel.dsms_on
            and tel.ts_active
        ):
            self._set_state(MacroState.MANUAL_DRIVING, now)
            return

        if (
            tel.ebs_status == EbsStatus.ARMED
            and tel.dsms_on
            and tel.dbs_check_ok
            and tel.ts_active
            and tel.dmi_mission in VALID_DRIVERLESS_MISSIONS
            and tel.brakes_engaged
            and tel.sdc_closed
        ):
            self._mission = tel.dmi_mission
            self._set_state(MacroState.DS_READY, now)

    def _step_ds_ready(self, tel: VcuTelemetryMessage, now: float) -> None:
        """Evaluates DS_READY faults, exit, and the RES Go start sequence.

        Checks run in priority order:

            1. Faults: an open SDC or an activated EBS trips DS_EMERGENCY.
            2. Clean exit: DSMS off with brakes released returns to DS_OFF.
            3. Lockout: RES Go is ignored for DS_READY_LOCKOUT_DURATION_S, and
               a press during the lockout is remembered as premature.
            4. Premature press: it must be released before a new press counts,
               so holding RES Go through the lockout cannot start the mission.
            5. Start: a fresh press enters DS_DRIVING if EBS is armed and DSMS,
               TS and the DBS check are still good.

        Args:
            tel: Latest VCU telemetry packet.
            now: Current telemetry timestamp.
        """
        if not tel.sdc_closed:
            self._trip_emergency(DiagnosticErrorCode.SDC_OPEN, now)
            return
        if tel.ebs_status == EbsStatus.ACTIVATED:
            self._trip_emergency(DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED, now)
            return
        if not tel.dsms_on and not tel.brakes_engaged:
            self._set_state(MacroState.DS_OFF, now)
            return

        # A press during the 5 s lockout must be released before a press counts.
        if now - self._ready_start_time < DS_READY_LOCKOUT_DURATION_S:
            self._premature_res_detected |= tel.res_go_signal
            return
        if self._premature_res_detected:
            self._premature_res_detected = tel.res_go_signal
            return

        if (
            tel.res_go_signal
            and tel.ebs_status == EbsStatus.ARMED
            and tel.dsms_on
            and tel.ts_active
            and tel.dbs_check_ok
        ):
            self._set_state(MacroState.DS_DRIVING, now)
            self._start_mission(now)

    def _step_ds_driving(
        self,
        tel: VcuTelemetryMessage,
        dt: float,
        now: float,
    ) -> Optional[ActuationCommandMessage]:
        """Runs Level 2 and evaluates DS_DRIVING faults and completion.

        Hardware faults trip first. Then the planner link produces a result:
        an e-stop trips DS_EMERGENCY, and "mission finished" enters DS_FINISHED
        once the car is at standstill, or trips STANDSTILL_TIMEOUT if it is
        still moving after FINISH_STANDSTILL_TIMEOUT_S.

        Args:
            tel: Latest VCU telemetry packet.
            dt: Seconds since the previous packet.
            now: Current telemetry timestamp.

        Returns:
            The requested actuation command, or None to stop the car.
        """
        hardware_faults = (
            (not tel.sdc_closed, DiagnosticErrorCode.SDC_OPEN),
            (tel.ebs_status == EbsStatus.ACTIVATED, DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED),
            (not tel.dsms_on, DiagnosticErrorCode.DSMS_UNEXPECTED_DROP),
            (not tel.ts_active, DiagnosticErrorCode.TS_INACTIVE),
        )
        for failed, fault in hardware_faults:
            if failed:
                self._trip_emergency(fault, now)
                return None

        result: SubStateResult | DiagnosticErrorCode = self._planner_link.tick(now=now)
        if isinstance(result, DiagnosticErrorCode):  # Level 2 timed out
            self._trip_emergency(result, now)
            return None

        if result.next_substate:
            self._substate = result.next_substate
        if result.emergency_stop:
            self._trip_emergency(DiagnosticErrorCode.LEVEL2_EMERGENCY_STOP, now)
            return None

        # If mission still in progress, return actuation command.
        if not result.mission_finished:
            self._finish_requested_time = None
            return result.actuation_command

        if self._finish_requested_time is None:
            self._finish_requested_time = now
        if self._standstill.is_standstill():
            self._set_state(MacroState.DS_FINISHED, now)
            return None
        if now - self._finish_requested_time > FINISH_STANDSTILL_TIMEOUT_S:
            self._trip_emergency(DiagnosticErrorCode.STANDSTILL_TIMEOUT, now)
            return None
        return result.actuation_command

    def _step_ds_finished(self, tel: VcuTelemetryMessage, now: float) -> None:
        """Returns to DS_OFF once DSMS is off and the brakes are released.

        Args:
            tel: Latest VCU telemetry packet.
            now: Current telemetry timestamp.
        """
        if not tel.dsms_on and not tel.brakes_engaged:
            self._set_state(MacroState.DS_OFF, now)

    def _step_ds_emergency(self, tel: VcuTelemetryMessage, now: float) -> None:
        """Returns to DS_OFF once EBS sound, DSMS and brakes are all off.

        Args:
            tel: Latest VCU telemetry packet.
            now: Current telemetry timestamp.
        """
        # Never clear an emergency the VCU has not been told about yet.
        if not self._emergency_reported:
            return
        if not tel.ebs_sound_active and not tel.dsms_on and not tel.brakes_engaged:
            logger.info("EMERGENCY CLEARED at t=%.3fs (faults were %r)", now, self._error_code)
            self._set_state(MacroState.DS_OFF, now)

    def _step_manual_driving(self, tel: VcuTelemetryMessage, now: float) -> None:
        """Returns to DS_OFF when the tractive system turns off.

        Args:
            tel: Latest VCU telemetry packet.
            now: Current telemetry timestamp.
        """
        if not tel.ts_active:
            self._set_state(MacroState.DS_OFF, now)

    # -----------------------------------------------------------------------
    # Level 2 helpers
    # -----------------------------------------------------------------------

    def receive_planner_message(self, msg: Union[ActuationCommandMessage, Level2StatusMessage]) -> None:
        """Passes message down to the planner link.
        
        Passing down the message through a helper keeps the planner link
        invisible to the StateMachineNode. This should hopefully help
        with encapsulation.
        
        Args:
            msg: Message to pass to planner link.
        """
        self._planner_link.receive(msg)
    
    def _start_mission(self, now: float) -> None:
        """Initializes mission related variables and starts the planner link.
        
        Args:
            now: Timestamp of start.
        """
        self._finish_requested_time = None
        self._standstill.reset()
        self._substate = "INIT"
        self._planner_link.start(now)

    def _stop_mission(self) -> None:
        """Stops the current mission in the planner link."""
        self._planner_link.stop()

    # -----------------------------------------------------------------------
    # Transitions and outputs
    # -----------------------------------------------------------------------

    def _set_state(self, new_state: MacroState, now: float) -> None:
        """Performs a transition and its entry and exit side effects.

        Entering DS_READY starts the lockout timer. Entering DS_OFF clears 
        faults and the committed mission. A transition to the current state 
        does nothing.

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

        if old_state == MacroState.DS_DRIVING:
            self._stop_mission()
        if new_state == MacroState.DS_READY:
            self._ready_start_time = now
            self._premature_res_detected = False
        elif new_state == MacroState.DS_OFF:
            self._error_code = DiagnosticErrorCode.NONE
            self._mission = Mission.NOT_SELECTED
            self._substate = "NONE"

    def _trip_emergency(self, fault: DiagnosticErrorCode, now: float) -> None:
        """Latches a fault, logs it with the running mission, and enters DS_EMERGENCY.

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
        self._emergency_reported = False
        self._set_state(MacroState.DS_EMERGENCY, now)

    def _outputs(self, now: float, raw_cmd: Optional[ActuationCommandMessage]) -> Outputs:
        """Builds the status message and the gated actuation command.

        Args:
            now: Current telemetry timestamp.
            raw_cmd: Requested actuation, or None to stop the car.

        Returns:
            A (StateMachineStatusMessage, ActuationCommandMessage) pair.
        """
        self._heartbeat = (self._heartbeat + 1) & HEARTBEAT_MAX
        if self._state == MacroState.DS_EMERGENCY:
            self._emergency_reported = True
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
