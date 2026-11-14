"""Shared fixtures and helper factories for State Machine unit tests."""

from typing import Optional, Tuple

import numpy as np
import pytest

from Pipeline.messages import ActuationCommandMessage, EbsStatus, MacroState, Mission, VcuTelemetryMessage
from StateMachine.constants import STATE_MACHINE_DEFAULT_DT_S
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.level2.registry import MissionRegistry

DT = STATE_MACHINE_DEFAULT_DT_S


@pytest.fixture(autouse=True)
def _isolate_mission_registry():
    """Undo any MissionRegistry.register() a test makes, so tests cannot leak missions."""
    saved = dict(MissionRegistry._registry)
    yield
    MissionRegistry._registry.clear()
    MissionRegistry._registry.update(saved)


@pytest.fixture
def state_machine() -> Level1StateMachine:
    """Fixture providing a fresh Level 1 state machine instance."""
    return Level1StateMachine()


def make_telemetry(timestamp: float = 0.0, velocity: float = 0.0, **overrides) -> VcuTelemetryMessage:
    """Quiescent AS_Off telemetry (car parked, everything off), with field overrides."""
    fields = dict(
        timestamp=timestamp,
        ebs_status=EbsStatus.ARMED,
        ebs_sound_active=False,
        dsms_status=False,
        dbs_check_ok=True,
        ts_status=False,
        ami_mission=Mission.NOT_SELECTED,
        brake_status=False,
        sdc_status=True,
        res_go_signal=False,
        velocity=np.array([velocity, 0.0, 0.0]),
    )
    fields.update(overrides)
    return VcuTelemetryMessage(**fields)


@pytest.fixture
def base_telemetry() -> VcuTelemetryMessage:
    """Fixture providing a default quiescent telemetry message in AS_Off."""
    return make_telemetry()


def make_ready_telemetry(
    timestamp: float = 0.0,
    res_go: bool = False,
    mission: Mission = Mission.INSPECTION,
    velocity: float = 0.0,
    **overrides,
) -> VcuTelemetryMessage:
    """Telemetry that satisfies every AS_Off -> AS_Ready precondition."""
    fields = dict(
        dsms_status=True,
        ts_status=True,
        ami_mission=mission,
        brake_status=True,
        res_go_signal=res_go,
    )
    fields.update(overrides)
    return make_telemetry(timestamp=timestamp, velocity=velocity, **fields)


def drive_to_as_driving(
    sm: Level1StateMachine,
    mission: Mission = Mission.INSPECTION,
    start: float = 1.0,
    command: Optional[ActuationCommandMessage] = None,
) -> float:
    """Walk ``sm`` through AS_Ready and the 5 s lockout into AS_Driving. Returns the last timestamp."""
    t = start
    sm.step(make_ready_telemetry(timestamp=t, mission=mission))
    for _ in range(260):
        t += DT
        sm.step(make_ready_telemetry(timestamp=t, mission=mission))
    t += DT
    if command is not None:
        command.timestamp = t
    sm.step(make_ready_telemetry(timestamp=t, res_go=True, mission=mission), external_actuation=command)
    assert sm.current_state == MacroState.AS_DRIVING
    return t


def step_until(sm, make_tel, start: float, state: MacroState, max_steps: int = 1000) -> Tuple[float, object]:
    """Step with ``make_tel(t)`` until ``sm`` reaches ``state``. Returns (t, last status)."""
    t, status = start, None
    for _ in range(max_steps):
        t += DT
        status, _ = sm.step(make_tel(t))
        if status.current_state == state:
            break
    return t, status
