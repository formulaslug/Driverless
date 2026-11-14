"""AS_Driving -> AS_Finished / AS_Emergency paths for registered and external-planner missions."""

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
from StateMachine.level1_state_machine import EXTERNAL_PLANNER_SUBSTATE, Level1StateMachine
from StateMachine.level2.base import MissionBehavior, MissionContext, SubStateResult
from StateMachine.tests.conftest import DT, drive_to_as_driving, make_ready_telemetry, step_until


def brake_cmd(t: float) -> ActuationCommandMessage:
    return ActuationCommandMessage(timestamp=t, braking=np.array([1.0]))


def planner_status(t: float, finished: bool = False, estop: bool = False) -> Level2StatusMessage:
    return Level2StatusMessage(timestamp=t, active_substate="BRAKING", mission_finished=finished, emergency_stop=estop)


class FinishedMission(MissionBehavior):
    """Registered-style mission that reports finished on every tick."""

    @property
    def name(self) -> str:
        return "Always finished"

    def on_enter(self, context: MissionContext) -> None:
        pass

    def update(self, context: MissionContext) -> SubStateResult:
        return SubStateResult(next_substate="DONE", mission_finished=True)

    def on_exit(self, context: MissionContext, interrupted: bool = False) -> None:
        pass


def test_external_planner_mission_reaches_finished() -> None:
    """flowchart.png: Mission Finished & velocity = 0 -> AS_Finished, for a mission with no registered behavior."""
    sm = Level1StateMachine()
    t = drive_to_as_driving(sm, Mission.ACCELERATION, command=brake_cmd(0.0))
    assert sm.active_substate_name == EXTERNAL_PLANNER_SUBSTATE

    status = None
    for _ in range(100):
        t += DT
        tel = make_ready_telemetry(timestamp=t, mission=Mission.ACCELERATION)
        status, _ = sm.step(tel, external_actuation=brake_cmd(t), external_status=planner_status(t, finished=True))
        if status.current_state == MacroState.AS_FINISHED:
            break
    assert status.current_state == MacroState.AS_FINISHED
    assert status.ebs_trigger_cmd is True


def test_external_status_from_before_mission_start_is_ignored() -> None:
    """A 'finished' status left over from a previous run must not finish the new one."""
    sm = Level1StateMachine()
    t = drive_to_as_driving(sm, Mission.ACCELERATION, command=brake_cmd(0.0))
    old_status = planner_status(t - 10.0, finished=True)
    for _ in range(100):
        t += DT
        tel = make_ready_telemetry(timestamp=t, mission=Mission.ACCELERATION)
        status, _ = sm.step(tel, external_actuation=brake_cmd(t), external_status=old_status)
    assert status.current_state == MacroState.AS_DRIVING


def test_external_planner_emergency_stop() -> None:
    sm = Level1StateMachine()
    t = drive_to_as_driving(sm, Mission.ACCELERATION, command=brake_cmd(0.0))
    t += DT
    tel = make_ready_telemetry(timestamp=t, mission=Mission.ACCELERATION)
    status, _ = sm.step(tel, external_actuation=brake_cmd(t), external_status=planner_status(t, estop=True))
    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.error_code == DiagnosticErrorCode.LEVEL2_EMERGENCY_STOP


def test_finished_while_moving_does_not_finish() -> None:
    """Negative case for Driving -> Finished: velocity must be 0, not just 'mission finished'."""
    sm = Level1StateMachine(mission_behavior=FinishedMission())
    t = drive_to_as_driving(sm)
    for _ in range(int(FINISH_STANDSTILL_TIMEOUT_S / DT) - 5):
        t += DT
        status, _ = sm.step(make_ready_telemetry(timestamp=t, velocity=1.0))
        assert status.current_state == MacroState.AS_DRIVING


def test_finished_but_never_stops_trips_standstill_timeout() -> None:
    sm = Level1StateMachine(mission_behavior=FinishedMission())
    t = drive_to_as_driving(sm)
    _, status = step_until(
        sm, lambda t: make_ready_telemetry(timestamp=t, velocity=1.0), t, MacroState.AS_EMERGENCY, max_steps=400
    )
    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.error_code == DiagnosticErrorCode.STANDSTILL_TIMEOUT


def test_status_reports_committed_mission_not_selector() -> None:
    """Changing the AMI selector mid-run does not change what the status says is running."""
    sm = Level1StateMachine()
    t = drive_to_as_driving(sm, Mission.INSPECTION)
    status, _ = sm.step(make_ready_telemetry(timestamp=t + DT, mission=Mission.SKIDPAD))
    assert status.active_mission == Mission.INSPECTION
    assert sm.active_mission == Mission.INSPECTION


def test_emergency_log_names_the_running_mission(caplog: pytest.LogCaptureFixture) -> None:
    sm = Level1StateMachine()
    t = drive_to_as_driving(sm, Mission.ACCELERATION, command=brake_cmd(0.0))
    tel = make_ready_telemetry(timestamp=t + DT, mission=Mission.ACCELERATION, sdc_status=False)
    with caplog.at_level(logging.INFO):
        sm.step(tel, external_actuation=brake_cmd(t + DT))
    trip = [r.message for r in caplog.records if "EMERGENCY TRIP" in r.message]
    assert trip and "mission=ACCELERATION" in trip[0]


def test_finished_ignores_telemetry_gap() -> None:
    """flowchart.png has no AS_Finished -> AS_Emergency arrow; a late packet does not trip it."""
    sm = Level1StateMachine(mission_behavior=FinishedMission())
    t = drive_to_as_driving(sm)
    t, status = step_until(sm, lambda t: make_ready_telemetry(timestamp=t), t, MacroState.AS_FINISHED)
    assert status.current_state == MacroState.AS_FINISHED
    status, _ = sm.step(make_ready_telemetry(timestamp=t + 0.5))
    assert status.current_state == MacroState.AS_FINISHED
    assert sm.handle_telemetry_timeout(t + 1.0) is None
