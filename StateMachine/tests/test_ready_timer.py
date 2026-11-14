"""Unit tests verifying the 5-second AS_Ready lockout timer and premature RES Go rejection."""

import pytest

from Pipeline.messages import MacroState, DSSIStatus
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.tests.conftest import make_ready_telemetry


def test_lockout_timer_duration(state_machine: Level1StateMachine) -> None:
    """Car remains in AS_Ready during the entire 5.0 second lockout window."""
    t = 10.0
    # Enter AS_Ready at t = 10.0
    status, _ = state_machine.step(make_ready_telemetry(timestamp=t))
    assert status.current_state == MacroState.AS_READY

    # Step through 4.9 seconds
    for _ in range(245):
        t += 0.02
        status, _ = state_machine.step(make_ready_telemetry(timestamp=t, res_go=False))
        assert status.current_state == MacroState.AS_READY
        assert state_machine.ready_timer_elapsed < 5.0

    # Step past 5.0s lockout to 5.2s total (260 steps * 0.02s = 5.2s)
    for _ in range(15):
        t += 0.02
        status, _ = state_machine.step(make_ready_telemetry(timestamp=t, res_go=False))
        assert status.current_state == MacroState.AS_READY

    assert state_machine.ready_timer_elapsed >= 5.0


def test_premature_res_go_rejection(state_machine: Level1StateMachine) -> None:
    """RES Go received before 5.0s is rejected and does NOT launch vehicle when 5.0s elapses."""
    t = 0.0
    state_machine.step(make_ready_telemetry(timestamp=t))
    assert state_machine.current_state == MacroState.AS_READY

    # Advance to t = 2.0s
    for _ in range(100):
        t += 0.02
        state_machine.step(make_ready_telemetry(timestamp=t))

    # Press RES Go prematurely at t = 2.02s
    t += 0.02
    status, _ = state_machine.step(make_ready_telemetry(timestamp=t, res_go=True))
    assert status.current_state == MacroState.AS_READY, "Premature RES Go improperly triggered transition!"

    # Release RES Go and advance past 5.0s lockout to t = 6.0s
    for _ in range(200):
        t += 0.02
        status, _ = state_machine.step(make_ready_telemetry(timestamp=t, res_go=False))
        assert status.current_state == MacroState.AS_READY, (
            "Car auto-launched at 5.0s due to remembered premature RES Go!"
        )

    # Fresh RES Go press at t = 6.02s triggers AS_Driving
    t += 0.02
    status, _ = state_machine.step(make_ready_telemetry(timestamp=t, res_go=True))
    assert status.current_state == MacroState.AS_DRIVING
    assert status.dssi_status == DSSIStatus.FLASHING_YELLOW


def test_held_res_go_across_boundary_is_rejected(state_machine: Level1StateMachine) -> None:
    """Holding RES Go continuously from before 5.0s through 5.0s is rejected until re-pressed."""
    t = 0.0
    state_machine.step(make_ready_telemetry(timestamp=t))
    assert state_machine.current_state == MacroState.AS_READY

    # Advance to t = 4.0s
    for _ in range(200):
        t += 0.02
        state_machine.step(make_ready_telemetry(timestamp=t))

    # Press and hold RES Go from t = 4.02s through t = 5.5s
    for _ in range(75):
        t += 0.02
        status, _ = state_machine.step(make_ready_telemetry(timestamp=t, res_go=True))
        assert status.current_state == MacroState.AS_READY, (
            "Car launched on continuous button hold across 5s boundary!"
        )

    # Release button at t = 5.52s
    t += 0.02
    status, _ = state_machine.step(make_ready_telemetry(timestamp=t, res_go=False))
    assert status.current_state == MacroState.AS_READY

    # Re-press fresh at t = 5.54s -> now launches into AS_Driving
    t += 0.02
    status, _ = state_machine.step(make_ready_telemetry(timestamp=t, res_go=True))
    assert status.current_state == MacroState.AS_DRIVING


def test_valid_fresh_res_go_at_five_seconds(state_machine: Level1StateMachine) -> None:
    """A fresh RES Go pressed at exactly t >= 5.0s transitions immediately to AS_Driving."""
    t = 0.0
    state_machine.step(make_ready_telemetry(timestamp=t))

    # Step exactly 5.0s with no button press
    for _ in range(250):
        t += 0.02
        state_machine.step(make_ready_telemetry(timestamp=t, res_go=False))

    # First button press at t = 5.02s
    t += 0.02
    status, _ = state_machine.step(make_ready_telemetry(timestamp=t, res_go=True))
    assert status.current_state == MacroState.AS_DRIVING
