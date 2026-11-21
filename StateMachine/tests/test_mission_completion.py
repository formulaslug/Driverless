"""DS_Driving -> DS_Finished / DS_Emergency paths for registered and external-planner missions."""

import logging

import numpy as np
import pytest

from Pipeline.messages import (
    ActuationCommandMessage,
    DiagnosticErrorCode,
    Level2StatusMessage,
    MacroState,
    Mission,
)
from StateMachine.constants import FINISH_STANDSTILL_TIMEOUT_S
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.level2.planner_link import PlannerLink
from StateMachine.tests.conftest import DT, drive_to_ds_driving, make_ready_telemetry, step_until, planner_step


def brake_cmd(t: float) -> ActuationCommandMessage:
    return ActuationCommandMessage(timestamp=t, braking=np.array([1.0]))


def planner_status(t: float, finished: bool = False, estop: bool = False) -> Level2StatusMessage:
    return Level2StatusMessage(timestamp=t, active_substate="BRAKING", mission_finished=finished, emergency_stop=estop)


def test_mission_reaches_finished() -> None:
    """flowchart.png: Mission Finished & velocity = 0 -> DS_Finished"""
    sm = Level1StateMachine(planner_link=PlannerLink())
    t = drive_to_ds_driving(sm, Mission.ACCELERATION)
    assert sm.active_substate_name == "INIT"

    status = None
    for _ in range(100):
        t += DT
        tel = make_ready_telemetry(timestamp=t, mission=Mission.ACCELERATION)
        status, _ = planner_step(sm, tel, finished=True, substate="BRAKING")
        assert status.active_substate == "BRAKING"
        if status.current_state == MacroState.DS_FINISHED:
            break
    assert status.current_state == MacroState.DS_FINISHED
    assert status.ebs_trigger_cmd is True


def test_status_from_before_mission_start_is_ignored() -> None:
    """A 'finished' status left over from a previous run must not finish the new one."""
    sm = Level1StateMachine(planner_link=PlannerLink())
    t = drive_to_ds_driving(sm, Mission.ACCELERATION)
    old_status = planner_status(t - 10.0, finished=True)
    for _ in range(100):
        t += DT
        tel = make_ready_telemetry(timestamp=t, mission=Mission.ACCELERATION)
        status, _ = planner_step(sm, tel)
    assert status.current_state == MacroState.DS_DRIVING


def test_planner_emergency_stop() -> None:
    sm = Level1StateMachine(planner_link=PlannerLink())
    t = drive_to_ds_driving(sm, Mission.ACCELERATION)
    t += DT
    tel = make_ready_telemetry(timestamp=t, mission=Mission.ACCELERATION)
    status, _ = planner_step(sm=sm, tel=tel, estop=True)
    assert status.current_state == MacroState.DS_EMERGENCY
    assert status.error_code == DiagnosticErrorCode.LEVEL2_EMERGENCY_STOP


def test_finished_while_moving_does_not_finish() -> None:
    """Negative case for Driving -> Finished: velocity must be 0, not just 'mission finished'."""
    sm = Level1StateMachine(planner_link=PlannerLink())
    t = drive_to_ds_driving(sm)
    for _ in range(int(FINISH_STANDSTILL_TIMEOUT_S / DT) - 5):
        t += DT
        tel = make_ready_telemetry(timestamp=t, velocity=1.0)
        status, _ = planner_step(sm=sm, tel=tel, finished=True)
        assert status.current_state == MacroState.DS_DRIVING


def test_finished_but_never_stops_trips_standstill_timeout() -> None:
    """Planner reports finished but the car keeps moving -> STANDSTILL_TIMEOUT."""
    sm = Level1StateMachine(planner_link=PlannerLink())
    t = drive_to_ds_driving(sm)
    _, status = step_until(
        sm,
        lambda t: make_ready_telemetry(timestamp=t, velocity=1.0),
        t,
        MacroState.DS_EMERGENCY,
        max_steps=400,
        planner=lambda t: dict(finished=True),
    )
    assert status.current_state == MacroState.DS_EMERGENCY
    assert status.error_code == DiagnosticErrorCode.STANDSTILL_TIMEOUT


def test_status_reports_committed_mission_not_selector() -> None:
    """Changing the DMI selector mid-run does not change what the status says is running."""
    sm = Level1StateMachine(planner_link=PlannerLink())
    t = drive_to_ds_driving(sm, Mission.INSPECTION)
    sm.receive_planner_message(ActuationCommandMessage(timestamp=t))
    status, _ = sm.step(make_ready_telemetry(timestamp=t + DT, mission=Mission.SKIDPAD))
    assert status.active_mission == Mission.INSPECTION
    assert sm.active_mission == Mission.INSPECTION


def test_emergency_log_names_the_running_mission(caplog: pytest.LogCaptureFixture) -> None:
    sm = Level1StateMachine(planner_link=PlannerLink())
    t = drive_to_ds_driving(sm, Mission.ACCELERATION)
    tel = make_ready_telemetry(timestamp=t + DT, mission=Mission.ACCELERATION, sdc_closed=False)
    with caplog.at_level(logging.INFO):
        sm.receive_planner_message(brake_cmd(t + DT))
        sm.step(tel)
    trip = [r.message for r in caplog.records if "EMERGENCY TRIP" in r.message]
    assert trip and "mission=ACCELERATION" in trip[0]


def test_finished_ignores_telemetry_gap() -> None:
    """flowchart.png has no DS_Finished -> DS_Emergency arrow; a late packet does not trip it."""
    sm = Level1StateMachine(planner_link=PlannerLink())
    t = drive_to_ds_driving(sm)
    t, status = step_until(
        sm,
        lambda t: make_ready_telemetry(timestamp=t),
        t,
        MacroState.DS_FINISHED,
        planner=lambda t: dict(finished=True),
    )
    assert status.current_state == MacroState.DS_FINISHED
    # No planner messages from here on: DS_Finished must not consult Level 2.
    status, _ = sm.step(make_ready_telemetry(timestamp=t + 0.5))
    assert status.current_state == MacroState.DS_FINISHED
    assert sm.handle_telemetry_timeout(t + 1.0) is None
