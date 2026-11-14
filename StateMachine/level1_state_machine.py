"""Deterministic Level 1 Macro-State Machine for FSAE Driverless."""

import logging
from typing import Optional, Tuple, Set, Dict
import numpy as np

from Pipeline.messages import (
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
    STATE_MACHINE_DEFAULT_DT_S,
    AS_READY_LOCKOUT_DURATION_S,
    VCU_WATCHDOG_TIMEOUT_S,
    LEVEL2_WATCHDOG_TIMEOUT_S,
    STANDSTILL_MIN_DURATION_S,
    STANDSTILL_VELOCITY_THRESHOLD_MPS,
    HEARTBEAT_MAX,
)
from StateMachine.standstill_detector import StandstillDetector
from StateMachine.gatekeeper import gate_command
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
    """Deterministic FSAE Driverless Level 1 macro-state machine.

    Governs the mandatory competition safety states, driven by VCU telemetry
    and gating Level 3 actuation commands from the active Level 2 mission:

        AS_OFF: System powered off / safe neutral.
        AS_READY: Preconditions verified, 5.0 s warning timer, service brake
            clamped.
        AS_DRIVING: Autonomous mission active, 50 Hz watchdogs, Level 3 gated.
        AS_FINISHED: Mission complete, verified standstill, EBS locked.
        AS_EMERGENCY: Safety circuit open or critical fault; latched until
            manual reset.
        MANUAL_DRIVING: TS active with manual mission selected and DSMS off.

    Typical usage example:

        state_machine = Level1StateMachine()
        status, actuation = state_machine.step(telemetry=tel, external_actuation=cmd)

    Args:
        mission_behavior: Optional Level 2 mission to run on entering
            AS_DRIVING. If None, the mission is created from the
            MissionRegistry using the telemetry's AMI mission selection.

    Attributes:
        current_state: The active MacroState.
        dssi_status: The DSSI light state for the active macro-state.
        error_code: DiagnosticErrorCode bitmask of the latched faults.
        heartbeat: Rolling counter incremented on each step, wrapping at
            HEARTBEAT_MAX.
        ready_timer_elapsed: Seconds spent in AS_READY, or 0.0 in any other
            state.
        active_mission: The Mission being executed, or None if no mission
            is active.
        active_substate_name: Name of the active Level 2 sub-state, or
            "NONE".
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
        """Trips the state machine immediately into AS_EMERGENCY.

        The trip is unconditional, since entering a safer state is never
        blocked. The fault is OR'ed into the latched error code, so earlier
        faults are preserved. If already in AS_EMERGENCY, the fault is still
        recorded but no transition occurs.

        Args:
            fault: Diagnostic fault flag(s) to latch into the error code.
            timestamp: Time of the fault in seconds, used as the AS_EMERGENCY
                entry time and for logging.
        """
        self._trip_emergency(fault, timestamp)

    def can_reset(self, telemetry: VcuTelemetryMessage) -> bool:
        """Checks whether telemetry satisfies all emergency reset preconditions.

        Per the FSAE rules, leaving AS_EMERGENCY requires EBS sound off,
        ASMS (DSMS) off, and brakes released. The acoustic warning buzzer is
        managed directly by VCU firmware and silences after the mandated
        countdown, so the CAN bridge expresses these conditions through the
        telemetry fields checked below.

        This method has no side effects; use request_reset() to perform the
        reset.

        Args:
            telemetry: Latest VCU telemetry snapshot.

        Returns:
            True if the state machine is in AS_EMERGENCY and all of the
            following hold: ebs_status is not ACTIVATED, dsms_status is
            False, brake_status is False, and sdc_status is True. False
            otherwise, including whenever the current state is not
            AS_EMERGENCY.
        """
        return (
            self._state == MacroState.AS_EMERGENCY
            and telemetry.ebs_status != EbsStatus.ACTIVATED
            and not telemetry.dsms_status
            and not telemetry.brake_status
            and telemetry.sdc_status
        )

    def request_reset(self, telemetry: VcuTelemetryMessage) -> bool:
        """Attempts to clear the AS_EMERGENCY latch and return to AS_OFF.

        On success, all latched faults are cleared and the AS_OFF entry
        actions run (brake and EBS commands released, active sub-state name
        cleared). On rejection, the state is unchanged and the reset-relevant
        telemetry fields (EBS, DSMS, brake, SDC) are logged as a warning.

        Args:
            telemetry: Latest VCU telemetry snapshot. Its timestamp is used as
                the AS_OFF entry time.

        Returns:
            True if the reset was accepted, False if any precondition checked
            by can_reset() was not met.
        """
        if self.can_reset(telemetry):
            logger.info("EMERGENCY RESET ACCEPTED at t=%.3fs: returning to AS_OFF", telemetry.timestamp)
            self._error_code = DiagnosticErrorCode.NONE
            self._set_state(MacroState.AS_OFF, telemetry, telemetry.timestamp)
            return True

        logger.warning(
            "EMERGENCY RESET REJECTED at t=%.3fs [ebs=%s, dsms=%s, brake=%s, sdc=%s]",
            telemetry.timestamp,
            telemetry.ebs_status,
            telemetry.dsms_status,
            telemetry.brake_status,
            telemetry.sdc_status,
        )
        return False

    # -----------------------------------------------------------------------
    # Main Step Cycle
    # -----------------------------------------------------------------------

    def step(
        self,
        telemetry: VcuTelemetryMessage,
        external_actuation: Optional[ActuationCommandMessage] = None,
    ) -> Tuple[StateMachineStatusMessage, ActuationCommandMessage]:
        """Executes one 50 Hz state machine cycle.

        Each call runs the following stages in order:

            1. Bookkeeping: advances the heartbeat and pushes the telemetry
               velocity into the standstill buffer.
            2. VCU watchdog: in AS_READY, AS_DRIVING, or AS_FINISHED, trips
               WATCHDOG_TIMEOUT_VCU if the gap since the previous call
               exceeds VCU_WATCHDOG_TIMEOUT_S.
            3. State logic: runs the handler for the current state, which may
               perform at most one transition and returns a candidate
               actuation command, i.e. the command requested before safety
               gating. Only AS_DRIVING produces a non-None candidate.
            4. Gating: passes the candidate through ActuationGatekeeper using
               the post-transition state, which clamps it, replaces it with a
               safe braking command, or discards it. A fault tripped during
               this cycle is therefore enforced on this cycle's output.

        All time is taken from telemetry.timestamp rather than the system
        clock, so replaying the same telemetry sequence is deterministic.
        Exceptions raised by the Level 2 mission's update() are caught and
        latched as LEVEL2_EXCEPTION rather than propagated; exceptions from
        its on_enter() are not (see _set_state()).

        Args:
            telemetry: Latest VCU telemetry snapshot. Timestamps must be
                monotonically increasing across calls. On the first call, dt
                defaults to STATE_MACHINE_DEFAULT_DT_S.
            external_actuation: Latest command from an external Level 2
                planner, or None. Only used in AS_DRIVING, where an internal
                mission's command takes precedence. With no internal mission,
                a missing command or one older than LEVEL2_WATCHDOG_TIMEOUT_S
                trips WATCHDOG_TIMEOUT_LEVEL2.

        Returns:
            A tuple (status, actuation) where status is the
            StateMachineStatusMessage describing the state, DSSI, latched
            faults, and brake/EBS commands after this cycle, and actuation is
            the gated ActuationCommandMessage that is safe to send to the VCU.
            actuation.gated is True only if a Level 2 command was passed
            through (clamped) in AS_DRIVING.
        """
        now = telemetry.timestamp
        dt = (now - self._last_tick_time) if self._last_tick_time is not None else STATE_MACHINE_DEFAULT_DT_S
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

        match self._state:
            case MacroState.AS_OFF:
                raw_cmd = self._step_as_off(telemetry, now)
            case MacroState.AS_READY:
                raw_cmd = self._step_as_ready(telemetry, now)
            case MacroState.AS_DRIVING:
                raw_cmd = self._step_as_driving(telemetry, dt, now, external_actuation)
            case MacroState.AS_FINISHED:
                raw_cmd = self._step_as_finished(telemetry, now)
            case MacroState.AS_EMERGENCY:
                raw_cmd = self._step_as_emergency(telemetry, now)
            case MacroState.MANUAL_DRIVING:
                raw_cmd = self._step_manual_driving(telemetry, now)

        # 3. Gated Actuation Filter
        gated_cmd = gate_command(
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
        """Evaluates AS_OFF exit conditions.

        MANUAL_DRIVING is checked first and takes priority: it is entered when
        the manual mission is selected, EBS is UNAVAILABLE, DSMS is off, and TS
        is active. Otherwise AS_READY is entered when EBS is ARMED, DSMS is
        on, the DBS check passed, TS is active, a mission in
        VALID_DRIVERLESS_MISSIONS is selected, the brakes are engaged, and
        SDC is closed.

        Args:
            tel: Latest VCU telemetry snapshot.
            now: Current time in seconds, from tel.timestamp.

        Returns:
            Always None, as no actuation is requested in AS_OFF.
        """
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
        """Evaluates AS_READY faults, exit, and the RES Go start sequence.

        Checks run in priority order:

            1. Faults: an open SDC trips SDC_OPEN, and an activated EBS trips
               EBS_ACTIVATED_UNEXPECTED.
            2. Clean exit: DSMS off with brakes released returns to AS_OFF.
            3. Lockout: RES Go is ignored until AS_READY_LOCKOUT_DURATION_S
               has elapsed since entry. A press during the lockout is
               remembered as premature.
            4. Premature press: after the lockout, a premature press must be
               released before a new press is accepted, so holding RES Go
               through the lockout cannot start the mission.
            5. Start: a fresh RES Go press enters AS_DRIVING if EBS is ARMED,
               DSMS is on, TS is active, SDC is closed, and the DBS check
               passed. If these do not all hold, the press is ignored and the
               state remains AS_READY.

        Args:
            tel: Latest VCU telemetry snapshot.
            now: Current time in seconds, from tel.timestamp.

        Returns:
            Always None. The service brake hold in AS_READY is applied by
            ActuationGatekeeper.
        """
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
        """Monitors faults, runs the Level 2 mission, and selects the command.

        Checks run in priority order, and any fault trips AS_EMERGENCY and
        returns None immediately:

            1. Hardware faults: open SDC (SDC_OPEN), activated EBS
               (EBS_ACTIVATED_UNEXPECTED), DSMS dropped (DSMS_UNEXPECTED_DROP),
               or TS inactive (TS_INACTIVE).
            2. Level 2 mission: with an active internal mission, calls its
               update() and applies the returned sub-state and command. An
               exception from update() trips LEVEL2_EXCEPTION. With no
               internal mission, a missing external command or one older
               than LEVEL2_WATCHDOG_TIMEOUT_S trips WATCHDOG_TIMEOUT_LEVEL2.
            3. Emergency stop: a mission result with emergency_stop set trips
               LEVEL2_EXCEPTION.
            4. Completion: a mission result with mission_finished set enters
               AS_FINISHED only once the standstill detector confirms speed
               below STANDSTILL_VELOCITY_THRESHOLD_MPS for at least
               STANDSTILL_MIN_DURATION_S. Until then, driving continues.

        Args:
            tel: Latest VCU telemetry snapshot.
            dt: Seconds since the previous step, passed to the mission.
            now: Current time in seconds, from tel.timestamp.
            external_cmd: Latest command from an external Level 2 planner,
                or None.

        Returns:
            The candidate actuation command: the internal mission's command
            if it returned one, otherwise external_cmd. None if a fault was
            tripped or AS_FINISHED was entered this cycle.
        """
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
        else:
            # Driving without internal mission: external planner must actively provide commands
            if external_cmd is None or (now - external_cmd.timestamp) > LEVEL2_WATCHDOG_TIMEOUT_S:
                self._trip_emergency(DiagnosticErrorCode.WATCHDOG_TIMEOUT_LEVEL2, now)
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
        """Returns to AS_OFF once DSMS is off and the brakes are released.

        No hardware fault checks run in this state; only the VCU watchdog in
        step() applies.

        Args:
            tel: Latest VCU telemetry snapshot.
            now: Current time in seconds, from tel.timestamp.

        Returns:
            Always None. The EBS hold in AS_FINISHED is applied by
            ActuationGatekeeper.
        """
        if (not tel.dsms_status) and (not tel.brake_status):
            self._set_state(MacroState.AS_OFF, tel, now)
        return None

    def _step_as_emergency(self, tel: VcuTelemetryMessage, now: float) -> Optional[ActuationCommandMessage]:
        """Holds the AS_EMERGENCY latch.

        No transition is evaluated here. The only exit is an accepted
        request_reset().

        Args:
            tel: Latest VCU telemetry snapshot (unused).
            now: Current time in seconds (unused).

        Returns:
            Always None. The EBS hold in AS_EMERGENCY is applied by
            ActuationGatekeeper.
        """
        return None

    def _step_manual_driving(self, tel: VcuTelemetryMessage, now: float) -> Optional[ActuationCommandMessage]:
        """Returns to AS_OFF once the tractive system is inactive.

        Args:
            tel: Latest VCU telemetry snapshot.
            now: Current time in seconds, from tel.timestamp.

        Returns:
            Always None, as the driver controls the vehicle directly.
        """
        if not tel.ts_status:
            self._set_state(MacroState.AS_OFF, tel, now)
        return None

    # -----------------------------------------------------------------------
    # Transition Logic
    # -----------------------------------------------------------------------

    def _set_state(self, new_state: MacroState, tel: VcuTelemetryMessage, now: float) -> None:
        """Performs a state transition and its exit and entry side effects.

        This is the only method that writes the state. A transition to the
        current state is a no-op. Otherwise, in order:

            1. Logs the transition and updates the state, entry time, and
               DSSI status.
            2. Leaving AS_DRIVING: calls the active mission's on_exit(), with
               interrupted=True if entering AS_EMERGENCY. Exceptions from
               on_exit() are logged and suppressed. On entering AS_OFF, the
               active mission is reset to the injected one, if any.
            3. Runs the entry actions for the new state:

                AS_OFF: releases EBS and service brake commands, clears the
                    sub-state name and premature RES flag.
                AS_READY: starts the ready timer, clears the premature RES
                    flag, engages the service brake.
                AS_DRIVING: starts the mission timer, releases brakes, resets
                    the standstill detector, then selects the injected
                    mission or creates one from MissionRegistry for the
                    selected AMI mission and calls its on_enter(). With no
                    mission, the sub-state name is set to "DRIVING_DEFAULT".
                AS_FINISHED: triggers EBS, releases the service brake.
                AS_EMERGENCY: triggers EBS and engages the service brake.
                MANUAL_DRIVING: releases EBS and service brake commands.

        Unlike on_exit(), exceptions from on_enter() are not caught and
        propagate out of step() with the state already set to AS_DRIVING.

        Args:
            new_state: State to transition to.
            tel: Telemetry used for logging, mission contexts, and mission
                selection on entering AS_DRIVING.
            now: Current time in seconds, used as the state entry time.
        """
        old_state = self._state
        if old_state == new_state:
            return

        mission_name = tel.ami_mission.name if hasattr(tel.ami_mission, "name") else str(tel.ami_mission)
        logger.info(
            "STATE TRANSITION: %s -> %s [t=%.3fs, mission=%s, substate=%s]",
            old_state.name,
            new_state.name,
            now,
            mission_name,
            self._active_substate_name,
        )

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
        """Latches the fault and transitions to AS_EMERGENCY.

        The fault is OR'ed into the error code before transitioning, so it is
        recorded even when already in AS_EMERGENCY (where the transition is a
        no-op). Logs the fault and accumulated bitmask at CRITICAL level.

        The transition uses a synthetic telemetry message (EBS ACTIVATED,
        DSMS off, all other fields at their defaults) rather than the real
        telemetry. The mission name in the transition log and the telemetry
        in the mission's on_exit() context therefore come from those
        defaults, not from the vehicle.

        Args:
            fault: Diagnostic fault flag(s) to latch.
            now: Current time in seconds, used as the AS_EMERGENCY entry time.
        """
        self._error_code |= int(fault)
        fault_name = fault.name if hasattr(fault, "name") else str(fault)
        logger.critical(
            "EMERGENCY TRIP LATCHED! Fault: %s (bitmask: 0x%04X, active_state: %s, t=%.3fs)",
            fault_name,
            self._error_code,
            self._state.name,
            now,
        )
        dummy_tel = VcuTelemetryMessage(timestamp=now, ebs_status=EbsStatus.ACTIVATED, dsms_status=False)
        self._set_state(MacroState.AS_EMERGENCY, dummy_tel, now)
