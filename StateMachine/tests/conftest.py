"""Shared fixtures and helper factories for State Machine unit tests."""

import pytest
import numpy as np

from Pipeline.messages import (
    EbsStatus,
    Mission,
    MacroState,
    DSSIStatus,
    DiagnosticErrorCode,
    VcuTelemetryMessage,
    ActuationCommandMessage,
)
from StateMachine.level1_state_machine import Level1StateMachine


@pytest.fixture
def state_machine() -> Level1StateMachine:
    """Fixture providing a fresh Level 1 state machine instance."""
    return Level1StateMachine()


@pytest.fixture
def base_telemetry() -> VcuTelemetryMessage:
    """Fixture providing a default quiescent telemetry message in AS_Off."""
    return VcuTelemetryMessage(
        timestamp=0.0,
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


def make_ready_telemetry(
    timestamp: float = 0.0,
    res_go: bool = False,
    mission: Mission = Mission.INSPECTION,
    velocity: float = 0.0,
) -> VcuTelemetryMessage:
    """Helper to create a telemetry packet that satisfies all AS_Ready preconditions."""
    return VcuTelemetryMessage(
        timestamp=timestamp,
        ebs_status=EbsStatus.ARMED,
        dsms_status=True,
        dbs_check_ok=True,
        ts_status=True,
        ami_mission=mission,
        brake_status=True,
        sdc_status=True,
        res_go_signal=res_go,
        velocity=np.array([velocity, 0.0, 0.0], dtype=np.float64),
    )
