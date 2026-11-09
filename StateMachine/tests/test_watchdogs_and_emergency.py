"""Unit tests verifying safety watchdogs, fault trips, and emergency latching."""

import pytest
import numpy as np

from messages import (
    MacroState,
    DSSIStatus,
    EbsStatus,
    Mission,
    DiagnosticErrorCode,
    VcuTelemetryMessage,
)
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.level2.base import MissionBehavior, MissionContext, SubStateResult
from StateMachine.level2.registry import MissionRegistry
from StateMachine.tests.conftest import make_ready_telemetry


def test_vcu_watchdog_timeout_trips_emergency(state_machine: Level1StateMachine) -> None:
    """Telemetry gap exceeding 200ms in AS_Ready trips VCU watchdog into AS_Emergency."""
    # Enter AS_Ready at t = 1.0
    tel = make_ready_telemetry(timestamp=1.0)
    state_machine.step(tel)
    assert state_machine.current_state == MacroState.AS_READY

    # Next packet arrives at t = 1.3s (300ms gap > 200ms timeout)
    tel.timestamp = 1.3
    status, act = state_machine.step(tel)

    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.dssi_status == DSSIStatus.CONTINUOUS_BLUE
    assert status.error_code & DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU
    assert status.error_flags[2]  # Bit 2 corresponds to WATCHDOG_TIMEOUT_VCU
    assert status.ebs_trigger_cmd is True
    assert act.gated is False


def test_sdc_open_trips_emergency(state_machine: Level1StateMachine) -> None:
    """SDC opening immediately trips to AS_Emergency with SDC_OPEN fault code."""
    tel = make_ready_telemetry(timestamp=1.0)
    state_machine.step(tel)
    assert state_machine.current_state == MacroState.AS_READY

    # Trip SDC
    tel.timestamp = 1.02
    tel.sdc_status = False
    status, _ = state_machine.step(tel)

    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.SDC_OPEN


def test_unexpected_ebs_activation_trips_emergency(state_machine: Level1StateMachine) -> None:
    """Unexpected EBS activation in AS_Ready immediately trips to AS_Emergency."""
    tel = make_ready_telemetry(timestamp=1.0)
    state_machine.step(tel)
    assert state_machine.current_state == MacroState.AS_READY

    # Trip EBS
    tel.timestamp = 1.02
    tel.ebs_status = EbsStatus.ACTIVATED
    status, _ = state_machine.step(tel)

    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED


def test_dsms_drop_in_driving_trips_emergency(state_machine: Level1StateMachine) -> None:
    """Switching DSMS off during AS_Driving trips to AS_Emergency."""
    t = 1.0
    state_machine.step(make_ready_telemetry(timestamp=t))

    for _ in range(260):
        t += 0.02
        state_machine.step(make_ready_telemetry(timestamp=t))

    # Launch driving
    t += 0.02
    state_machine.step(make_ready_telemetry(timestamp=t, res_go=True))
    assert state_machine.current_state == MacroState.AS_DRIVING

    # Drop DSMS
    t += 0.02
    tel = make_ready_telemetry(timestamp=t)
    tel.dsms_status = False
    status, _ = state_machine.step(tel)

    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.DSMS_UNEXPECTED_DROP


class CrashingMission(MissionBehavior):
    """Test mission that raises an unhandled exception."""
    @property
    def name(self) -> str:
        return "Crashing Mission"

    def on_enter(self, context: MissionContext) -> None:
        pass

    def update(self, context: MissionContext) -> SubStateResult:
        raise RuntimeError("Simulated unhandled controls fault")

    def on_exit(self, context: MissionContext, interrupted: bool = False) -> None:
        pass


def test_level2_exception_latches_emergency(state_machine: Level1StateMachine) -> None:
    """Unhandled exception in Level 2 update is safely caught and trips AS_Emergency."""
    # Register crashing mission for Skidpad
    MissionRegistry.register(Mission.SKIDPAD, CrashingMission)

    t = 1.0
    state_machine.step(make_ready_telemetry(timestamp=t, mission=Mission.SKIDPAD))

    for _ in range(260):
        t += 0.02
        state_machine.step(make_ready_telemetry(timestamp=t, mission=Mission.SKIDPAD))

    # Launch driving with crashing mission (transitions to AS_DRIVING)
    t += 0.02
    status, act = state_machine.step(make_ready_telemetry(timestamp=t, res_go=True, mission=Mission.SKIDPAD))
    assert status.current_state == MacroState.AS_DRIVING

    # Next driving tick calls Level 2 update(), which raises exception and trips AS_Emergency
    t += 0.02
    status, act = state_machine.step(make_ready_telemetry(timestamp=t, mission=Mission.SKIDPAD))
    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.LEVEL2_EXCEPTION
    assert act.gated is False


def test_emergency_latching_and_reset_guards(state_machine: Level1StateMachine) -> None:
    """AS_Emergency latches and cannot be reset until exact safety reset criteria are met."""
    tel = make_ready_telemetry(timestamp=1.0)
    state_machine.step(tel)
    state_machine.request_emergency_trip(DiagnosticErrorCode.SDC_OPEN, timestamp=1.02)
    assert state_machine.current_state == MacroState.AS_EMERGENCY

    # Regular step in emergency keeps it in emergency
    status, _ = state_machine.step(make_ready_telemetry(timestamp=1.04))
    assert status.current_state == MacroState.AS_EMERGENCY

    # Attempt reset with DSMS on -> Must fail
    tel_bad1 = make_ready_telemetry(timestamp=1.06)
    tel_bad1.dsms_status = True
    assert not state_machine.request_reset(tel_bad1)
    assert state_machine.current_state == MacroState.AS_EMERGENCY

    # Attempt reset with EBS activated -> Must fail
    tel_bad2 = make_ready_telemetry(timestamp=1.08)
    tel_bad2.dsms_status = False
    tel_bad2.ebs_status = EbsStatus.ACTIVATED
    assert not state_machine.request_reset(tel_bad2)
    assert state_machine.current_state == MacroState.AS_EMERGENCY

    # Attempt reset with brakes engaged -> Must fail
    tel_bad3 = make_ready_telemetry(timestamp=1.10)
    tel_bad3.dsms_status = False
    tel_bad3.brake_status = True
    assert not state_machine.request_reset(tel_bad3)
    assert state_machine.current_state == MacroState.AS_EMERGENCY

    # Valid reset: DSMS off, brakes released, EBS disarmed, SDC closed
    tel_valid = VcuTelemetryMessage(
        timestamp=1.12,
        ebs_status=EbsStatus.ARMED,
        dsms_status=False,
        dbs_check_ok=True,
        ts_status=False,
        ami_mission=Mission.NOT_SELECTED,
        brake_status=False,
        sdc_status=True,
        res_go_signal=False,
        velocity=np.zeros(3, dtype=np.float64),
    )
    assert state_machine.request_reset(tel_valid)
    assert state_machine.current_state == MacroState.AS_OFF
    assert state_machine.error_code == DiagnosticErrorCode.NONE
