"""Unit tests verifying safety watchdogs, fault trips, and emergency latching."""

import pytest
import numpy as np

from Pipeline.messages import (
    MacroState,
    DSSIStatus,
    EbsStatus,
    Mission,
    DiagnosticErrorCode,
    VcuTelemetryMessage,
    ActuationCommandMessage,
)
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.level2.base import MissionBehavior, MissionContext, SubStateResult
from StateMachine.level2.registry import MissionRegistry
from StateMachine.tests.conftest import make_ready_telemetry, make_telemetry


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


def test_injected_mission_behavior() -> None:
    """Verifies that a custom injected MissionBehavior overrides the default registry."""
    class InjectedMission(MissionBehavior):
        def __init__(self) -> None:
            self.entered: bool = False
            self.updated: bool = False

        @property
        def name(self) -> str:
            return "Injected Test Mission"

        def on_enter(self, context: MissionContext) -> None:
            self.entered = True

        def update(self, context: MissionContext) -> SubStateResult:
            self.updated = True
            return SubStateResult(next_substate="CUSTOM_SUBSTATE")

        def on_exit(self, context: MissionContext, interrupted: bool = False) -> None:
            pass

    custom_mission = InjectedMission()
    sm = Level1StateMachine(mission_behavior=custom_mission)

    t = 1.0
    sm.step(make_ready_telemetry(timestamp=t))

    for _ in range(260):
        t += 0.02
        sm.step(make_ready_telemetry(timestamp=t))

    t += 0.02
    status, _ = sm.step(make_ready_telemetry(timestamp=t, res_go=True))
    assert status.current_state == MacroState.AS_DRIVING
    assert custom_mission.entered is True

    t += 0.02
    status, _ = sm.step(make_ready_telemetry(timestamp=t))
    assert custom_mission.updated is True
    assert status.active_substate == "CUSTOM_SUBSTATE"


@pytest.mark.parametrize(
    "blocking",
    [
        {"ebs_sound_active": True},
        {"dsms_status": True},
        {"brake_status": True},
    ],
)
def test_emergency_holds_until_every_exit_condition(state_machine: Level1StateMachine, blocking: dict) -> None:
    """flowchart.png: AS_Emergency stays latched while any of EBS sound / ASMS / brakes is still on."""
    state_machine.step(make_ready_telemetry(timestamp=1.0))
    state_machine.request_emergency_trip(DiagnosticErrorCode.SDC_OPEN, timestamp=1.02)

    for i in range(50):
        status, _ = state_machine.step(make_telemetry(timestamp=1.04 + i * 0.02, **blocking))
        assert status.current_state == MacroState.AS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.SDC_OPEN


def test_emergency_exits_to_off_per_flowchart(state_machine: Level1StateMachine) -> None:
    """flowchart.png: EBS Sound Off & ASMS off & Brakes Released -> AS_Off, clearing the fault code.

    The SDC is still open here (normal right after an EBS trip); the flowchart does not require it closed.
    """
    state_machine.step(make_ready_telemetry(timestamp=1.0))
    state_machine.request_emergency_trip(DiagnosticErrorCode.SDC_OPEN, timestamp=1.02)

    status, _ = state_machine.step(make_telemetry(timestamp=1.04, sdc_status=False, ebs_status=EbsStatus.ACTIVATED))
    assert status.current_state == MacroState.AS_OFF
    assert status.error_code == DiagnosticErrorCode.NONE
    assert status.ebs_trigger_cmd is False


def test_level2_external_planner_watchdog_timeout() -> None:
    """When relying on external controls in AS_Driving, missing commands for > 200ms trip watchdog."""
    sm = Level1StateMachine()  # No injected mission, Mission.TRACKDRIVE has no registered behavior
    t = 1.0
    sm.step(make_ready_telemetry(timestamp=t, mission=Mission.TRACKDRIVE))

    for _ in range(260):
        t += 0.02
        sm.step(make_ready_telemetry(timestamp=t, mission=Mission.TRACKDRIVE))

    # Launch driving
    t += 0.02
    cmd = ActuationCommandMessage(timestamp=t, steering=np.zeros(1), throttle=np.array([0.2]))
    status, _ = sm.step(make_ready_telemetry(timestamp=t, res_go=True, mission=Mission.TRACKDRIVE), external_actuation=cmd)
    assert status.current_state == MacroState.AS_DRIVING

    # Telemetry streams at 50Hz, but external planner sends stale commands (> 200ms old)
    for _ in range(12):  # 12 * 0.02s = 240ms > 200ms timeout
        t += 0.02
        status, act = sm.step(
            make_ready_telemetry(timestamp=t, mission=Mission.TRACKDRIVE),
            external_actuation=cmd,  # Stale timestamp
        )

    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.WATCHDOG_TIMEOUT_LEVEL2
    assert act.gated is False
