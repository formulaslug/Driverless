"""Shared fixtures and helper factories for State Machine unit tests."""

import dataclasses
from typing import Callable, Optional, Tuple

import numpy as np
import pytest

from Pipeline.messages import (
    ActuationCommandMessage,                           
    EbsStatus, 
    MacroState, 
    Mission, 
    VcuTelemetryMessage, 
    Level2StatusMessage
    )
from StateMachine.constants import STATE_MACHINE_DEFAULT_DT_S
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.level2.planner_link import PlannerLink

DT = STATE_MACHINE_DEFAULT_DT_S

@pytest.fixture
def state_machine() -> Level1StateMachine:
    """Fixture providing a fresh Level 1 state machine instance."""
    return Level1StateMachine(planner_link=PlannerLink())


def make_telemetry(timestamp: float = 0.0, velocity: float = 0.0, **overrides) -> VcuTelemetryMessage:
    """Quiescent DS_Off telemetry (car parked, everything off), with field overrides."""
    fields = dict(
        timestamp=timestamp,
        ebs_status=EbsStatus.ARMED,
        ebs_sound_active=False,
        dsms_on=False,
        dbs_check_ok=True,
        ts_active=False,
        dmi_mission=Mission.NOT_SELECTED,
        brakes_engaged=False,
        sdc_closed=True,
        res_go_signal=False,
        velocity=np.array([velocity, 0.0, 0.0]),
    )
    fields.update(overrides)
    return VcuTelemetryMessage(**fields)


@pytest.fixture
def base_telemetry() -> VcuTelemetryMessage:
    """Fixture providing a default quiescent telemetry message in DS_Off."""
    return make_telemetry()


def make_ready_telemetry(
    timestamp: float = 0.0,
    res_go: bool = False,
    mission: Mission = Mission.INSPECTION,
    velocity: float = 0.0,
    **overrides,
) -> VcuTelemetryMessage:
    """Telemetry that satisfies every DS_Off -> DS_Ready precondition."""
    fields = dict(
        dsms_on=True,
        ts_active=True,
        dmi_mission=mission,
        brakes_engaged=True,
        res_go_signal=res_go,
    )
    fields.update(overrides)
    return make_telemetry(timestamp=timestamp, velocity=velocity, **fields)


def drive_to_ds_driving(
    sm: Level1StateMachine,
    mission: Mission = Mission.INSPECTION,
    start: float = 1.0
) -> float:
    """Walk ``sm`` through DS_Ready and the 5 s lockout into DS_Driving. Returns the last timestamp."""
    t = start
    sm.step(make_ready_telemetry(timestamp=t, mission=mission))
    for _ in range(260):
        t += DT
        sm.step(make_ready_telemetry(timestamp=t, mission=mission))
    t += DT
    sm.step(make_ready_telemetry(timestamp=t, res_go=True, mission=mission))
    assert sm.current_state == MacroState.DS_DRIVING
    return t

def planner_step(sm: Level1StateMachine, tel, finished=False, estop=False, substate="DRIVING"):
    """One tick with a fake planner"""
    sm.receive_planner_message(ActuationCommandMessage(timestamp=tel.timestamp))
    sm.receive_planner_message(Level2StatusMessage(timestamp=tel.timestamp,
                                                   active_substate=substate,
                                                   mission_finished=finished,
                                                   emergency_stop=estop))
    return sm.step(tel)

PlannerFn = Callable[[float], Optional[dict]]


def step_until(
    sm: Level1StateMachine,
    make_tel: Callable[[float], VcuTelemetryMessage],
    start: float,
    state: MacroState,
    max_steps: int = 1000,
    planner: Optional[PlannerFn] = None,
) -> Tuple[float, object]:
    """Steps ``sm`` until it reaches ``state``.

    Args:
        sm: State machine to drive.
        make_tel: Builds the telemetry packet for time t.
        start: Timestamp before the first step; each step advances by DT.
        state: MacroState to stop at.
        max_steps: Upper bound on steps taken.
        planner: Called with t each tick. Return planner_step kwargs to send
            a fresh command and status that tick, or None to stay silent.
            Omit for phases with no planner (DS_OFF, DS_READY).

    Returns:
        A (last timestamp, last status) pair.
    """
    t, status = start, None
    for _ in range(max_steps):
        t += DT
        tel = make_tel(t)
        planner_kwargs = planner(t) if planner is not None else None
        if planner_kwargs is None:
            status, _ = sm.step(tel)
        else:
            status, _ = planner_step(sm, tel, **planner_kwargs)
        if status.current_state == state:
            break
    return t, status
