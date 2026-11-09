"""Unit tests for Level 2 Inspection Mission and sub-state behavior."""

import pytest
import numpy as np

from messages import Mission, VcuTelemetryMessage, EbsStatus
from StateMachine.level2.base import MissionContext
from StateMachine.level2.registry import MissionRegistry
from StateMachine.level2.missions.inspection import InspectionMission, InspectionSubState


def test_inspection_mission_registry() -> None:
    """Inspection mission is registered in MissionRegistry under Mission.INSPECTION."""
    mission = MissionRegistry.create(Mission.INSPECTION)
    assert mission is not None
    assert isinstance(mission, InspectionMission)


def test_inspection_mission_lifecycle_and_substates() -> None:
    """Inspection mission cleanly cycles through steering sweep, throttle ramp, and braking."""
    mission = InspectionMission()
    t = 0.0

    def make_ctx(time_now: float, vx: float = 0.0) -> MissionContext:
        return MissionContext(
            telemetry=VcuTelemetryMessage(
                timestamp=time_now,
                ebs_status=EbsStatus.ARMED,
                dsms_status=True,
                dbs_check_ok=True,
                ts_status=True,
                ami_mission=Mission.INSPECTION,
                brake_status=False,
                sdc_status=True,
                res_go_signal=False,
                velocity=np.array([vx, 0.0, 0.0], dtype=np.float64),
            ),
            dt=0.02,
            mission_time=time_now,
            current_substate=mission.current_substate.value,
        )

    # on_enter starts in STEERING_SWEEP
    mission.on_enter(make_ctx(t))
    assert mission.current_substate == InspectionSubState.STEERING_SWEEP

    # Test sinusoidal steering in sweep
    res = mission.update(make_ctx(1.0))
    assert not res.mission_finished
    assert res.actuation_command is not None
    # At t = 1.0, f = 0.5Hz -> sin(2 * pi * 0.5 * 1.0) = sin(pi) ~ 0
    # At t = 0.5, f = 0.5Hz -> sin(2 * pi * 0.5 * 0.5) = sin(pi/2) = 1.0 -> steer = A
    res_quarter = mission.update(make_ctx(0.5))
    assert np.isclose(res_quarter.actuation_command.steering[0], InspectionMission.STEER_AMPLITUDE_RAD)
    assert np.allclose(res_quarter.actuation_command.throttle, 0.0)

    # Step to 4.1s -> transitions to POWERTRAIN_RAMP
    res_ramp = mission.update(make_ctx(4.1))
    assert mission.current_substate == InspectionSubState.POWERTRAIN_RAMP

    # In ramp, verify throttle is positive and steering is zero
    res_ramp_mid = mission.update(make_ctx(6.0))
    assert res_ramp_mid.actuation_command.throttle[0] > 0.0
    assert np.allclose(res_ramp_mid.actuation_command.steering, 0.0)

    # Step to 8.2s -> transitions to CONTROLLED_STOP
    res_stop = mission.update(make_ctx(8.2, vx=1.0))
    assert mission.current_substate == InspectionSubState.CONTROLLED_STOP
    assert res_stop.actuation_command.braking[0] > 0.0
    assert not res_stop.mission_finished

    # Vehicle comes to standstill (vx = 0.0) during braking
    res_stop_still = mission.update(make_ctx(9.0, vx=0.0))
    assert mission.current_substate == InspectionSubState.COMPLETE
    assert res_stop_still.mission_finished is True
