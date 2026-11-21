"""State machine messages are immutable snapshots, including their arrays."""

import dataclasses

import numpy as np
import pytest

from Pipeline.messages import ActuationCommandMessage, Level2StatusMessage
from StateMachine.level2.base import SubStateResult
from StateMachine.tests.conftest import make_telemetry


def test_fields_cannot_be_reassigned() -> None:
    tel = make_telemetry(timestamp=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tel.timestamp = 2.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        SubStateResult().mission_finished = True


def test_arrays_cannot_be_edited_in_place() -> None:
    cmd = ActuationCommandMessage(timestamp=0.0, steering=0.1)
    with pytest.raises(ValueError):
        cmd.steering[0] = 0.4
    with pytest.raises(ValueError):
        make_telemetry(velocity=1.0).velocity[0] = 0.0
    status = Level2StatusMessage(timestamp=0.0, active_substate="X", mission_finished=False, emergency_stop=False)
    with pytest.raises(ValueError):
        status.fault_flags[0] = True


def test_caller_array_is_copied_not_locked() -> None:
    """The message keeps its own copy, so the sender can keep reusing its buffer."""
    buffer = np.array([0.2])
    cmd = ActuationCommandMessage(timestamp=0.0, steering=buffer)
    buffer[0] = 0.9
    assert cmd.steering[0] == 0.2
    assert buffer.flags.writeable


def test_replace_builds_a_new_message() -> None:
    tel = make_telemetry(timestamp=1.0)
    later = dataclasses.replace(tel, timestamp=1.02)
    assert tel.timestamp == 1.0 and later.timestamp == 1.02
    assert not later.velocity.flags.writeable
