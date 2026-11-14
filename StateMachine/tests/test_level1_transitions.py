"""Unit tests verifying all Level 1 macro-state transitions and guards."""

import pytest
import numpy as np

from Pipeline.messages import (
    MacroState,
    DSSIStatus,
    EbsStatus,
    Mission,
    DiagnosticErrorCode,
    VcuTelemetryMessage,
)
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.tests.conftest import make_ready_telemetry


def test_initial_state_is_as_off(state_machine: Level1StateMachine, base_telemetry: VcuTelemetryMessage) -> None:
    """System initializes in AS_Off with DSSI Off and no error codes."""
    status, act = state_machine.step(base_telemetry)
    assert status.current_state == MacroState.AS_OFF
    assert status.dssi_status == DSSIStatus.OFF
    assert status.error_code == DiagnosticErrorCode.NONE
    assert not status.ebs_trigger_cmd
    assert not status.service_brake_cmd


def test_transition_as_off_to_as_ready(state_machine: Level1StateMachine) -> None:
    """Validates transition from AS_Off to AS_Ready when all preconditions hold."""
    tel = make_ready_telemetry(timestamp=1.0)
    status, act = state_machine.step(tel)

    assert status.current_state == MacroState.AS_READY
    assert status.dssi_status == DSSIStatus.CONTINUOUS_YELLOW
    assert status.service_brake_cmd is True
    assert status.ebs_trigger_cmd is False


@pytest.mark.parametrize(
    "missing_condition_patch",
    [
        {"ebs_status": EbsStatus.ACTIVATED},
        {"ebs_status": EbsStatus.UNAVAILABLE},
        {"dsms_status": False},
        {"dbs_check_ok": False},
        {"ts_status": False},
        {"ami_mission": Mission.NOT_SELECTED},
        {"brake_status": False},
        {"sdc_status": False},
    ],
)
def test_as_off_stays_in_off_if_any_precondition_missing(
    state_machine: Level1StateMachine, missing_condition_patch: dict
) -> None:
    """Missing any single readiness precondition prevents transition to AS_Ready."""
    params = {
        "timestamp": 1.0,
        "ebs_status": EbsStatus.ARMED,
        "ebs_sound_active": False,
        "dsms_status": True,
        "dbs_check_ok": True,
        "ts_status": True,
        "ami_mission": Mission.INSPECTION,
        "brake_status": True,
        "sdc_status": True,
        "res_go_signal": False,
        "velocity": np.zeros(3, dtype=np.float64),
    }
    params.update(missing_condition_patch)
    tel = VcuTelemetryMessage(**params)

    status, _ = state_machine.step(tel)
    assert status.current_state == MacroState.AS_OFF
    assert status.dssi_status == DSSIStatus.OFF


def test_manual_driving_transition(state_machine: Level1StateMachine) -> None:
    """Transition to and from Manual Driving mode."""
    # Manual conditions: Manual mission & EBS unavailable & DSMS off & TS active
    manual_tel = VcuTelemetryMessage(
        timestamp=1.0,
        ebs_status=EbsStatus.UNAVAILABLE,
        ebs_sound_active=False,
        dsms_status=False,
        dbs_check_ok=True,
        ts_status=True,
        ami_mission=Mission.MANUAL,
        brake_status=False,
        sdc_status=True,
        res_go_signal=False,
        velocity=np.zeros(3, dtype=np.float64),
    )
    status, _ = state_machine.step(manual_tel)
    assert status.current_state == MacroState.MANUAL_DRIVING
    assert status.dssi_status == DSSIStatus.OFF

    # When TS turns off, return to AS_Off
    manual_tel.timestamp = 1.02
    manual_tel.ts_status = False
    status, _ = state_machine.step(manual_tel)
    assert status.current_state == MacroState.AS_OFF


def test_transition_as_ready_to_as_off(state_machine: Level1StateMachine) -> None:
    """AS_Ready transitions cleanly back to AS_Off when DSMS is switched off and brakes released."""
    tel = make_ready_telemetry(timestamp=1.0)
    status, _ = state_machine.step(tel)
    assert status.current_state == MacroState.AS_READY

    # Turn DSMS off and release brakes at next cycle
    tel.timestamp = 1.02
    tel.dsms_status = False
    tel.brake_status = False
    status, _ = state_machine.step(tel)
    assert status.current_state == MacroState.AS_OFF
    assert status.dssi_status == DSSIStatus.OFF


def test_cannot_skip_ready_straight_to_driving(state_machine: Level1StateMachine) -> None:
    """Sending RES Go while in AS_Off cannot bypass AS_Ready to jump to AS_Driving."""
    tel = make_ready_telemetry(timestamp=1.0, res_go=True)
    status, _ = state_machine.step(tel)
    # The car enters AS_Ready, but MUST NOT be in AS_Driving
    assert status.current_state == MacroState.AS_READY
    assert status.current_state != MacroState.AS_DRIVING


def test_transition_as_finished_to_as_off(state_machine: Level1StateMachine) -> None:
    """From AS_Finished, turning DSMS off and releasing brakes transitions to AS_Off."""
    # Enter AS_Ready
    t = 1.0
    state_machine.step(make_ready_telemetry(timestamp=t))

    # Advance past 5.0s lockout
    for _ in range(260):
        t += 0.02
        state_machine.step(make_ready_telemetry(timestamp=t))

    # Enter AS_Driving
    t += 0.02
    state_machine.step(make_ready_telemetry(timestamp=t, res_go=True))
    assert state_machine.current_state == MacroState.AS_DRIVING

    # Advance inspection mission until complete and standstill detected
    for _ in range(600):
        t += 0.02
        status, _ = state_machine.step(make_ready_telemetry(timestamp=t, velocity=0.0))
        if status.current_state == MacroState.AS_FINISHED:
            break

    assert state_machine.current_state == MacroState.AS_FINISHED
    assert state_machine.dssi_status == DSSIStatus.FLASHING_BLUE

    # Exit to AS_Off
    t += 0.02
    exit_tel = VcuTelemetryMessage(
        timestamp=t,
        ebs_status=EbsStatus.ACTIVATED,
        ebs_sound_active=False,
        dsms_status=False,
        dbs_check_ok=True,
        ts_status=False,
        ami_mission=Mission.NOT_SELECTED,
        brake_status=False,
        sdc_status=True,
        res_go_signal=False,
        velocity=np.zeros(3, dtype=np.float64),
    )
    status, _ = state_machine.step(exit_tel)
    assert status.current_state == MacroState.AS_OFF
    assert status.dssi_status == DSSIStatus.OFF


def test_state_transition_and_emergency_logging(state_machine: Level1StateMachine, caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that state transitions and emergency faults emit structured log messages."""
    import logging

    with caplog.at_level(logging.INFO):
        # Trigger AS_Off -> AS_Ready
        tel = make_ready_telemetry(timestamp=1.0)
        state_machine.step(tel)

        assert any("STATE TRANSITION: AS_OFF -> AS_READY" in record.message for record in caplog.records)

        # Trigger emergency trip
        state_machine.request_emergency_trip(DiagnosticErrorCode.SDC_OPEN, timestamp=1.02)
        assert any("EMERGENCY TRIP: SDC_OPEN" in record.message for record in caplog.records)


def test_telemetry_safety_fields_are_required() -> None:
    """A packet missing safety fields must fail to construct, not default to 'ready'."""
    with pytest.raises(TypeError):
        VcuTelemetryMessage(timestamp=0.0, ebs_status=EbsStatus.ARMED, dsms_status=True)
