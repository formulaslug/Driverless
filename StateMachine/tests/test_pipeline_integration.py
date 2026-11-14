"""Integration tests verifying StateMachineNode within the Pipeline pub/sub system."""

import time
import pytest
import numpy as np

from Pipeline.graph import Graph
from Pipeline.node import Node
from Pipeline.messages import (
    MacroState,
    DSSIStatus,
    EbsStatus,
    VcuTelemetryMessage,
    StateMachineStatusMessage,
    ActuationCommandMessage,
)
from StateMachine.state_machine_node import StateMachineNode
from StateMachine.mock_vcu import MockVcuNode, MockVcuSimulator


class TelemetrySinkNode(Node):
    """Sink node collecting outbound messages from StateMachineNode."""

    def __init__(self, name: str = "sink") -> None:
        super().__init__(name, queueSize=32)
        self.received_statuses: list = []
        self.received_actuations: list = []

    def process(self, message):
        if isinstance(message, StateMachineStatusMessage):
            self.received_statuses.append(message)
        elif isinstance(message, ActuationCommandMessage):
            self.received_actuations.append(message)
        return None


def test_statemachine_node_in_pipeline_graph() -> None:
    """Run MockVcuNode -> StateMachineNode -> TelemetrySinkNode inside Graph."""
    sim = MockVcuSimulator()
    sim.set_readiness_preconditions()

    vcu_node = MockVcuNode(name="mock_vcu", simulator=sim, tick_rate_hz=100.0, max_ticks=10)
    sm_node = StateMachineNode(name="state_machine")
    sink = TelemetrySinkNode(name="sink")

    vcu_node.subscribe(sm_node)
    sm_node.subscribe(sink)

    graph = Graph()
    graph.add(vcu_node)
    graph.add(sm_node)
    graph.add(sink)

    # Run pipeline graph to completion
    graph.start()
    graph.join()

    # Verify messages were received through the concurrent pipeline
    assert len(sink.received_statuses) == 10
    assert len(sink.received_actuations) == 10

    # First packet transitions from AS_Off to AS_Ready
    assert sink.received_statuses[0].current_state == MacroState.AS_READY
    assert sink.received_actuations[0].gated is False
    assert np.allclose(sink.received_actuations[0].throttle, 0.0)


def test_statemachine_node_processes_external_actuation() -> None:
    """StateMachineNode caches external actuation messages and passes them through when driving."""
    node = StateMachineNode()

    # Send external actuation request
    ext_cmd = ActuationCommandMessage(
        timestamp=1.0,
        steering=np.array([0.2], dtype=np.float64),
        throttle=np.array([0.7], dtype=np.float64),
        braking=np.array([0.0], dtype=np.float64),
    )
    result = node.process(ext_cmd)
    # Caching returns None
    assert result is None
    assert node._latest_external_actuation is ext_cmd


def test_statemachine_node_drops_stale_external_actuation() -> None:
    """StateMachineNode discards cached external actuation messages that exceed 200ms age."""
    node = StateMachineNode()

    # Cached command timestamp = 1.0s
    node.process(
        ActuationCommandMessage(
            timestamp=1.0,
            steering=np.array([0.1], dtype=np.float64),
            throttle=np.array([0.5], dtype=np.float64),
        )
    )
    assert node._latest_external_actuation is not None

    # Inbound telemetry at 1.3s (300ms later > 200ms timeout)
    tel = VcuTelemetryMessage(timestamp=1.3, ebs_status=EbsStatus.ARMED)
    res = node.process(tel)
    assert res is not None
    # Stale actuation was cleared
    assert node._latest_external_actuation is None


def test_statemachine_node_concurrent_access() -> None:
    """Verifies thread-safe concurrent actuation caching and telemetry stepping."""
    import threading

    node = StateMachineNode()
    sim = MockVcuSimulator()
    sim.set_readiness_preconditions()

    stop_event = threading.Event()
    errors = []

    def actuation_worker():
        t = 1.0
        while not stop_event.is_set():
            cmd = ActuationCommandMessage(
                timestamp=t,
                steering=np.array([0.1], dtype=np.float64),
                throttle=np.array([0.2], dtype=np.float64),
            )
            try:
                node.process(cmd)
            except Exception as e:
                errors.append(e)
            t += 0.01
            time.sleep(0.001)

    def telemetry_worker():
        t = 1.0
        while not stop_event.is_set():
            msg = sim.step(dt=0.01)
            try:
                node.process(msg)
            except Exception as e:
                errors.append(e)
            t += 0.01
            time.sleep(0.001)

    t1 = threading.Thread(target=actuation_worker)
    t2 = threading.Thread(target=telemetry_worker)
    t1.start()
    t2.start()

    time.sleep(0.05)
    stop_event.set()
    t1.join()
    t2.join()

    assert len(errors) == 0


def test_statemachine_node_request_reset() -> None:
    """Verifies StateMachineNode delegates emergency reset thread-safely."""
    from Pipeline.messages import DiagnosticErrorCode

    node = StateMachineNode()
    tel = VcuTelemetryMessage(timestamp=1.0, ebs_status=EbsStatus.ARMED, dsms_status=True)
    node.process(tel)

    node.state_machine.request_emergency_trip(DiagnosticErrorCode.SDC_OPEN, timestamp=1.02)
    assert node.state_machine.current_state == MacroState.AS_EMERGENCY

    # Reset with valid telemetry
    valid_reset_tel = VcuTelemetryMessage(
        timestamp=1.04,
        ebs_status=EbsStatus.ARMED,
        dsms_status=False,
        brake_status=False,
        sdc_status=True,
    )
    ok = node.request_reset(valid_reset_tel)
    assert ok is True
    assert node.state_machine.current_state == MacroState.AS_OFF


def test_mock_vcu_interactive_mode() -> None:
    """Verifies that run_interactive_mode responds to commands and controls state machine."""
    from StateMachine.mock_vcu import run_interactive_mode

    commands = [
        "help",
        "ready",
        "step 10",
        "speed 5.0",
        "dsms off",
        "dsms on",
        "ebs act",
        "status",
        "quit",
    ]
    outputs = []

    def mock_input(prompt: str = "") -> str:
        if commands:
            return commands.pop(0)
        return "quit"

    def mock_output(msg: str) -> None:
        outputs.append(msg)

    run_interactive_mode(input_fn=mock_input, output_fn=mock_output)

    assert any("Applied AS_Ready preconditions" in out for out in outputs)
    assert any("Stepped 10 cycles" in out for out in outputs)
    assert any("Longitudinal speed set to: 5.00 m/s" in out for out in outputs)
    assert any("Exiting interactive shell" in out for out in outputs)
