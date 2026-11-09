"""Unit tests for Level 3 Actuation Safety Gatekeeper and Multiplexer."""

import pytest
import numpy as np

from messages import MacroState, ActuationCommandMessage
from StateMachine.gatekeeper import ActuationGatekeeper


def test_gatekeeper_pass_through_in_as_driving() -> None:
    """In AS_Driving, requested actuation arrays pass through untouched with gated=True."""
    req = ActuationCommandMessage(
        timestamp=10.0,
        steering=np.array([0.15], dtype=np.float64),
        throttle=np.array([0.45], dtype=np.float64),
        braking=np.array([0.0], dtype=np.float64),
        gated=False,
    )

    gated = ActuationGatekeeper.gate_command(MacroState.AS_DRIVING, req, timestamp=10.0)
    assert gated.gated is True
    assert np.isclose(gated.steering[0], 0.15)
    assert np.isclose(gated.throttle[0], 0.45)
    assert np.isclose(gated.braking[0], 0.0)


@pytest.mark.parametrize(
    "non_driving_state,expected_brake",
    [
        (MacroState.AS_OFF, 0.0),
        (MacroState.AS_READY, 1.0),
        (MacroState.AS_FINISHED, 1.0),
        (MacroState.AS_EMERGENCY, 1.0),
        (MacroState.MANUAL_DRIVING, 0.0),
    ],
)
def test_gatekeeper_zeroing_in_non_driving_states(
    non_driving_state: MacroState, expected_brake: float
) -> None:
    """In all non-driving states, throttle and steering are hard-zeroed and appropriate braking applied."""
    # Attempt to command full throttle and turning outside AS_Driving
    malicious_cmd = ActuationCommandMessage(
        timestamp=5.0,
        steering=np.array([0.5], dtype=np.float64),
        throttle=np.array([1.0], dtype=np.float64),
        braking=np.array([0.0], dtype=np.float64),
        gated=True,
    )

    gated = ActuationGatekeeper.gate_command(non_driving_state, malicious_cmd, timestamp=5.0)
    assert gated.gated is False
    assert np.allclose(gated.throttle, 0.0)
    assert np.allclose(gated.steering, 0.0)
    assert np.allclose(gated.braking, expected_brake)


def test_gatekeeper_handles_none_command() -> None:
    """Gatekeeper handles None requested_command gracefully, producing safe defaults."""
    gated = ActuationGatekeeper.gate_command(MacroState.AS_READY, None, timestamp=1.0)
    assert gated.gated is False
    assert np.allclose(gated.throttle, 0.0)
    assert np.allclose(gated.steering, 0.0)
    assert np.allclose(gated.braking, 1.0)


def test_gatekeeper_trajectory_shape_preservation() -> None:
    """Gatekeeper preserves trajectory horizon dimensions when zeroing commands."""
    horizon_req = ActuationCommandMessage(
        timestamp=2.0,
        steering=np.ones((10, 1), dtype=np.float64),
        throttle=np.ones((10, 1), dtype=np.float64),
        braking=np.zeros((10, 1), dtype=np.float64),
        gated=True,
    )

    gated = ActuationGatekeeper.gate_command(MacroState.AS_EMERGENCY, horizon_req, timestamp=2.0)
    assert gated.gated is False
    assert gated.steering.shape == (10, 1)
    assert np.allclose(gated.steering, 0.0)
    assert np.allclose(gated.throttle, 0.0)
    assert np.allclose(gated.braking, 1.0)
