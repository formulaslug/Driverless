"""Unit tests for Level 3 Actuation Safety Gatekeeper and Multiplexer."""

import pytest
import numpy as np

from Pipeline.messages import MacroState, ActuationCommandMessage
from StateMachine.gatekeeper import gate_command


def test_gatekeeper_pass_through_in_as_driving() -> None:
    """In AS_Driving, requested actuation arrays pass through untouched with gated=True."""
    req = ActuationCommandMessage(
        timestamp=10.0,
        steering=np.array([0.15], dtype=np.float64),
        throttle=np.array([0.45], dtype=np.float64),
        braking=np.array([0.0], dtype=np.float64),
        gated=False,
    )

    gated = gate_command(MacroState.AS_DRIVING, req, timestamp=10.0)
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

    gated = gate_command(non_driving_state, malicious_cmd, timestamp=5.0)
    assert gated.gated is False
    assert np.allclose(gated.throttle, 0.0)
    assert np.allclose(gated.steering, 0.0)
    assert np.allclose(gated.braking, expected_brake)


def test_gatekeeper_handles_none_command() -> None:
    """Gatekeeper handles None requested_command gracefully, producing safe defaults."""
    gated = gate_command(MacroState.AS_READY, None, timestamp=1.0)
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

    gated = gate_command(MacroState.AS_EMERGENCY, horizon_req, timestamp=2.0)
    assert gated.gated is False
    assert gated.steering.shape == (10, 1)
    assert np.allclose(gated.steering, 0.0)
    assert np.allclose(gated.throttle, 0.0)
    assert np.allclose(gated.braking, 1.0)


def test_gatekeeper_rejects_nan_and_inf_commands() -> None:
    """Gatekeeper detects NaN or Inf in requested commands and clamps safe service brakes."""
    # Command containing NaN throttle
    nan_cmd = ActuationCommandMessage(
        timestamp=1.0,
        steering=np.array([0.1], dtype=np.float64),
        throttle=np.array([np.nan], dtype=np.float64),
        braking=np.array([0.0], dtype=np.float64),
    )
    gated_nan = gate_command(MacroState.AS_DRIVING, nan_cmd, timestamp=1.0)
    assert gated_nan.gated is False
    assert np.allclose(gated_nan.throttle, 0.0)
    assert np.allclose(gated_nan.steering, 0.0)
    assert np.allclose(gated_nan.braking, 1.0)

    # Command containing Inf steering
    inf_cmd = ActuationCommandMessage(
        timestamp=1.0,
        steering=np.array([np.inf], dtype=np.float64),
        throttle=np.array([0.5], dtype=np.float64),
        braking=np.array([0.0], dtype=np.float64),
    )
    gated_inf = gate_command(MacroState.AS_DRIVING, inf_cmd, timestamp=1.0)
    assert gated_inf.gated is False
    assert np.allclose(gated_inf.throttle, 0.0)
    assert np.allclose(gated_inf.steering, 0.0)
    assert np.allclose(gated_inf.braking, 1.0)


def test_gatekeeper_clamps_out_of_bounds_actuation() -> None:
    """Gatekeeper clamps requested commands to physical rack and normalized bounds."""
    excessive_cmd = ActuationCommandMessage(
        timestamp=2.0,
        steering=np.array([0.85], dtype=np.float64),   # Exceeds max rack limit (+25 deg ~ 0.4363 rad)
        throttle=np.array([1.80], dtype=np.float64),   # Exceeds 100%
        braking=np.array([-0.30], dtype=np.float64),   # Negative brake request
    )
    gated = gate_command(MacroState.AS_DRIVING, excessive_cmd, timestamp=2.0)
    assert gated.gated is True
    assert np.isclose(gated.steering[0], 0.4363)
    assert np.isclose(gated.throttle[0], 1.0)
    assert np.isclose(gated.braking[0], 0.0)


def test_gatekeeper_failsafe_on_none_in_driving() -> None:
    """If requested command is None in AS_Driving, fail-safe service brake is applied."""
    gated = gate_command(MacroState.AS_DRIVING, None, timestamp=3.0)
    assert gated.gated is False
    assert np.allclose(gated.throttle, 0.0)
    assert np.allclose(gated.steering, 0.0)
    assert np.allclose(gated.braking, 1.0)


def test_scalar_inputs_are_normalized_to_1d() -> None:
    """Python floats and 0-D arrays become shape (1,) so callers can always index [0]."""
    cmd = ActuationCommandMessage(timestamp=0.0, steering=0.1, throttle=np.float64(0.2), braking=0.0)
    gated = gate_command(MacroState.AS_DRIVING, cmd, timestamp=0.0)
    assert gated.gated is True
    assert gated.steering.shape == gated.throttle.shape == gated.braking.shape == (1,)
    assert np.isclose(gated.throttle[0], 0.2)


def test_mismatched_shapes_fall_back_to_service_brake() -> None:
    """A command whose channels disagree in shape is rejected, not passed through."""
    cmd = ActuationCommandMessage(timestamp=0.0, steering=np.zeros(3), throttle=np.ones(1), braking=np.zeros(1))
    gated = gate_command(MacroState.AS_DRIVING, cmd, timestamp=0.0)
    assert gated.gated is False
    assert np.allclose(gated.throttle, 0.0)
    assert np.allclose(gated.braking, 1.0)
