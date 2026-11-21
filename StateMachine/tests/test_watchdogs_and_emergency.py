"""Unit tests verifying safety watchdogs, fault trips, and emergency latching."""

import dataclasses

import pytest
import numpy as np

from Pipeline.messages import (
    MacroState,
    DSSIStatus,
    EbsStatus,
    Mission,
    DiagnosticErrorCode,
)
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.level2.planner_link import PlannerLink
from StateMachine.tests.conftest import DT, drive_to_ds_driving, make_ready_telemetry, make_telemetry, planner_step


def test_vcu_watchdog_timeout_trips_emergency(state_machine: Level1StateMachine) -> None:
    """Telemetry gap exceeding 200ms in DS_Ready trips VCU watchdog into DS_Emergency."""
    # Enter DS_Ready at t = 1.0
    tel = make_ready_telemetry(timestamp=1.0)
    planner_step(sm=state_machine, tel=tel)
    assert state_machine.current_state == MacroState.DS_READY

    # Next packet arrives at t = 1.3s (300ms gap > 200ms timeout)
    tel = dataclasses.replace(tel, timestamp=1.3)
    status, act = planner_step(sm=state_machine, tel=tel)

    assert status.current_state == MacroState.DS_EMERGENCY
    assert status.dssi_status == DSSIStatus.CONTINUOUS_BLUE
    assert status.error_code & DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU
    assert status.error_flags[2]  # Bit 2 corresponds to WATCHDOG_TIMEOUT_VCU
    assert status.ebs_trigger_cmd is True
    assert act.gated is False


def test_sdc_open_trips_emergency(state_machine: Level1StateMachine) -> None:
    """SDC opening immediately trips to DS_Emergency with SDC_OPEN fault code."""
    tel = make_ready_telemetry(timestamp=1.0)
    state_machine.step(tel)
    assert state_machine.current_state == MacroState.DS_READY

    # Trip SDC
    tel = dataclasses.replace(tel, timestamp=1.02, sdc_closed=False)
    status, _ = state_machine.step(tel)

    assert status.current_state == MacroState.DS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.SDC_OPEN


def test_unexpected_ebs_activation_trips_emergency(state_machine: Level1StateMachine) -> None:
    """Unexpected EBS activation in DS_Ready immediately trips to DS_Emergency."""
    tel = make_ready_telemetry(timestamp=1.0)
    state_machine.step(tel)
    assert state_machine.current_state == MacroState.DS_READY

    # Trip EBS
    tel = dataclasses.replace(tel, timestamp=1.02, ebs_status=EbsStatus.ACTIVATED)
    status, _ = state_machine.step(tel)

    assert status.current_state == MacroState.DS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED


def test_dsms_drop_in_driving_trips_emergency(state_machine: Level1StateMachine) -> None:
    """Switching DSMS off during DS_Driving trips to DS_Emergency."""
    t = 1.0
    state_machine.step(make_ready_telemetry(timestamp=t))

    for _ in range(260):
        t += 0.02
        state_machine.step(make_ready_telemetry(timestamp=t))

    # Launch driving
    t += 0.02
    state_machine.step(make_ready_telemetry(timestamp=t, res_go=True))
    assert state_machine.current_state == MacroState.DS_DRIVING

    # Drop DSMS
    t += 0.02
    tel = make_ready_telemetry(timestamp=t)
    tel = dataclasses.replace(tel, dsms_on=False)
    status, _ = state_machine.step(tel)

    assert status.current_state == MacroState.DS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.DSMS_UNEXPECTED_DROP


def test_level2_failure_latches_emergency(state_machine: Level1StateMachine) -> None:
    """A planner that crashes (goes silent) trips DS_Emergency, and recovering does not clear it."""
    t = drive_to_ds_driving(state_machine, Mission.SKIDPAD)

    # Healthy planner: commands pass through.
    for _ in range(5):
        t += DT
        status, act = planner_step(state_machine, make_ready_telemetry(timestamp=t, mission=Mission.SKIDPAD))
        assert status.current_state == MacroState.DS_DRIVING
        assert act.gated is True

    # Planner crashes: no messages for longer than LEVEL2_WATCHDOG_TIMEOUT_S.
    for _ in range(12):  # 12 * 20 ms = 240 ms > 200 ms
        t += DT
        status, act = state_machine.step(make_ready_telemetry(timestamp=t, mission=Mission.SKIDPAD))
    assert status.current_state == MacroState.DS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.WATCHDOG_TIMEOUT_LEVEL2
    assert act.gated is False

    # Planner restarts and sends fresh, healthy messages: the emergency must hold.
    for _ in range(50):
        t += DT
        status, act = planner_step(state_machine, make_ready_telemetry(timestamp=t, mission=Mission.SKIDPAD))
        assert status.current_state == MacroState.DS_EMERGENCY
        assert status.error_code & DiagnosticErrorCode.WATCHDOG_TIMEOUT_LEVEL2
        assert act.gated is False

@pytest.mark.parametrize(
    "blocking",
    [
        {"ebs_sound_active": True},
        {"dsms_on": True},
        {"brakes_engaged": True},
    ],
)
def test_emergency_holds_until_every_exit_condition(state_machine: Level1StateMachine, blocking: dict) -> None:
    """flowchart.png: DS_Emergency stays latched while any of EBS sound / DSMS / brakes is still on."""
    state_machine.step(make_ready_telemetry(timestamp=1.0))
    state_machine.request_emergency_trip(DiagnosticErrorCode.SDC_OPEN, timestamp=1.02)

    for i in range(50):
        status, _ = planner_step(state_machine, make_telemetry(timestamp=1.04 + i * 0.02, **blocking))
        assert status.current_state == MacroState.DS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.SDC_OPEN


def test_emergency_exits_to_off_per_flowchart(state_machine: Level1StateMachine) -> None:
    """flowchart.png: EBS Sound Off & DSMS off & Brakes Released -> DS_Off, clearing the fault code.

    The SDC is still open here (normal right after an EBS trip); the flowchart does not require it closed.
    """
    state_machine.step(make_ready_telemetry(timestamp=1.0))
    tripped = make_ready_telemetry(timestamp=1.02, sdc_closed=False)
    status, _ = state_machine.step(tripped)
    assert status.current_state == MacroState.DS_EMERGENCY

    status, _ = state_machine.step(make_telemetry(timestamp=1.04, sdc_closed=False, ebs_status=EbsStatus.ACTIVATED))
    assert status.current_state == MacroState.DS_OFF
    assert status.error_code == DiagnosticErrorCode.NONE
    assert status.ebs_trigger_cmd is False


def test_level2_planner_watchdog_timeout() -> None:
    """When relying on external controls in DS_Driving, missing commands for > 200ms trip watchdog."""
    sm = Level1StateMachine(planner_link=PlannerLink())  # No injected mission, Mission.TRACKDRIVE has no registered behavior
    t = 1.0
    planner_step(sm=sm, tel=make_ready_telemetry(timestamp=t, mission=Mission.TRACKDRIVE))

    for _ in range(260):
        t += 0.02
        planner_step(sm=sm, tel=make_ready_telemetry(timestamp=t, mission=Mission.TRACKDRIVE))

    # Launch driving
    t += 0.02
    status, _ = planner_step(sm=sm, tel=make_ready_telemetry(timestamp=t, res_go=True, mission=Mission.TRACKDRIVE))
    assert status.current_state == MacroState.DS_DRIVING

    # Telemetry streams at 50Hz, but external planner sends stale commands (> 200ms old)
    for _ in range(12):  # 12 * 0.02s = 240ms > 200ms timeout
        t += 0.02

        status, act = sm.step(
            make_ready_telemetry(timestamp=t, mission=Mission.TRACKDRIVE)  # Status and Actuation Commands left stale
        )

    assert status.current_state == MacroState.DS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.WATCHDOG_TIMEOUT_LEVEL2
    assert act.gated is False


@pytest.mark.parametrize(
    "fault_patch, expected_fault",
    [
        ({"ebs_status": EbsStatus.ACTIVATED}, DiagnosticErrorCode.EBS_ACTIVATED_UNEXPECTED),  # flowchart arrow
        ({"sdc_closed": False}, DiagnosticErrorCode.SDC_OPEN),
        ({"dsms_on": False}, DiagnosticErrorCode.DSMS_UNEXPECTED_DROP),
        ({"ts_active": False}, DiagnosticErrorCode.TS_INACTIVE),
    ],
)
def test_driving_hardware_fault_trips_emergency(fault_patch: dict, expected_fault: DiagnosticErrorCode) -> None:
    """DS_Driving -> DS_Emergency on EBS activation (flowchart.png) and the hardware safety extensions."""
    sm = Level1StateMachine(planner_link=PlannerLink())
    t = drive_to_ds_driving(sm)
    status, act = sm.step(make_ready_telemetry(timestamp=t + 0.02, **fault_patch))
    assert status.current_state == MacroState.DS_EMERGENCY
    assert status.error_code == expected_fault
    assert status.ebs_trigger_cmd is True
    assert act.gated is False and np.allclose(act.throttle, 0.0)

def test_emergency_is_published_before_it_can_clear() -> None:
    """A trip must reach the VCU at least once, even if the late packet already meets the exit conditions."""
    sm = Level1StateMachine(planner_link=PlannerLink())
    t = drive_to_ds_driving(sm)
    late_packet = make_telemetry(timestamp=t + 0.3)  # DSMS off, brakes released, sound off
    status, act = sm.step(late_packet)
    assert status.current_state == MacroState.DS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU
    assert status.ebs_trigger_cmd is True and np.allclose(act.braking, 1.0)


def test_external_trip_is_reported_before_it_can_clear(state_machine: Level1StateMachine) -> None:
    """request_emergency_trip() between ticks: the next packet reports DS_EMERGENCY instead of clearing it."""
    state_machine.step(make_ready_telemetry(timestamp=1.0))
    state_machine.request_emergency_trip(DiagnosticErrorCode.SDC_OPEN, timestamp=1.01)
    status, _ = state_machine.step(make_telemetry(timestamp=1.02))
    assert status.current_state == MacroState.DS_EMERGENCY
    status, _ = state_machine.step(make_telemetry(timestamp=1.04))
    assert status.current_state == MacroState.DS_OFF
