"""Integration tests verifying StateMachineNode within the Pipeline pub/sub system."""

import time
import pytest
import numpy as np

from graph import Graph
from node import Node
from messages import (
    MacroState,
    ASSIStatus,
    VcuTelemetryMessage,
    StateMachineStatusMessage,
    ActuationCommandMessage,
)
from StateMachine.node import StateMachineNode
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
