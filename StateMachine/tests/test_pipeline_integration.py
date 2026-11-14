"""Integration tests verifying StateMachineNode within the Pipeline pub/sub system."""

import queue
import time

import numpy as np

from Pipeline.graph import Graph
from Pipeline.node import Node, POISON
from Pipeline.messages import (
    ActuationCommandMessage,
    DiagnosticErrorCode,
    Level2StatusMessage,
    MacroState,
    Mission,
    StateMachineStatusMessage,
)
from StateMachine.state_machine_node import StateMachineNode
from StateMachine.mock_vcu import MockVcuNode, MockVcuSimulator
from StateMachine.tests.conftest import DT, make_ready_telemetry, make_telemetry


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


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_link_loss_trips_emergency_without_new_packets() -> None:
    """The VCU watchdog fires while the link is still dead, not only when a late packet arrives."""
    clock = FakeClock()
    node = StateMachineNode(clock=clock)
    node.process(make_ready_telemetry(timestamp=1.0))
    assert node.state_machine.current_state == MacroState.AS_READY

    clock.now += 0.1
    assert node.check_link() is None

    clock.now += 0.2  # 300 ms of silence
    status, actuation = node.check_link()
    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.error_code & DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU
    assert np.isclose(status.timestamp, 1.3)
    assert np.allclose(actuation.braking, 1.0) and actuation.gated is False

    clock.now += 1.0
    assert node.check_link() is None  # already in AS_Emergency: emit once, not every tick


def test_link_loss_is_ignored_in_as_off() -> None:
    clock = FakeClock()
    node = StateMachineNode(clock=clock)
    node.process(make_telemetry(timestamp=1.0))
    clock.now += 5.0
    assert node.check_link() is None
    assert node.state_machine.current_state == MacroState.AS_OFF


def test_watchdog_thread_publishes_emergency() -> None:
    """End to end with real threads: one packet, then silence, then an emergency is published."""
    node = StateMachineNode()
    out: queue.Queue = queue.Queue()
    node.subscribers.append(out)
    node.start()
    node.inputQueue.put(make_ready_telemetry(timestamp=1.0))

    states = []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and MacroState.AS_EMERGENCY not in states:
        try:
            message = out.get(timeout=0.1)
        except queue.Empty:
            continue
        if isinstance(message, StateMachineStatusMessage):
            states.append(message.current_state)

    node.inputQueue.put(POISON)
    node.join()
    assert states[0] == MacroState.AS_READY
    assert MacroState.AS_EMERGENCY in states


def test_node_forwards_planner_messages_to_state_machine() -> None:
    """Cached planner commands and status reach the state machine, so external missions can finish."""
    node = StateMachineNode()
    t = 1.0
    node.process(make_ready_telemetry(timestamp=t, mission=Mission.ACCELERATION))
    for _ in range(261):
        t += DT
        node.process(make_ready_telemetry(timestamp=t, mission=Mission.ACCELERATION))
    t += DT
    node.process(ActuationCommandMessage(timestamp=t, braking=np.array([1.0])))
    node.process(make_ready_telemetry(timestamp=t, res_go=True, mission=Mission.ACCELERATION))
    assert node.state_machine.current_state == MacroState.AS_DRIVING

    for _ in range(60):
        t += DT
        assert node.process(ActuationCommandMessage(timestamp=t, braking=np.array([1.0]))) is None
        assert node.process(
            Level2StatusMessage(timestamp=t, active_substate="STOP", mission_finished=True, emergency_stop=False)
        ) is None
        status, _ = node.process(make_ready_telemetry(timestamp=t, mission=Mission.ACCELERATION))
    assert status.current_state == MacroState.AS_FINISHED


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


def test_mock_vcu_interactive_reaches_manual_driving_and_emergency_exit() -> None:
    """Every flowchart state is reachable from the shell, including Manual Driving and the emergency exit."""
    from StateMachine.mock_vcu import run_interactive_mode

    commands = iter([
        "ebs unavail", "dsms off", "ts on", "mission manual",  # -> MANUAL_DRIVING
        "ts off",                                              # -> AS_OFF
        "ready", "sdc trip", "sound on",                       # AS_READY -> AS_EMERGENCY
        "sdc close", "dsms off", "brakes off", "sound off",    # -> AS_OFF
        "quit",
    ])
    outputs = []
    run_interactive_mode(input_fn=lambda _prompt: next(commands), output_fn=outputs.append)

    states = [line.split("State: ")[1].split(" |")[0] for line in outputs if "State: " in line]
    assert "MANUAL_DRIVING" in states
    i = states.index("AS_EMERGENCY")
    assert states[i:-1].count("AS_EMERGENCY") == len(states[i:-1])  # latched until the sound goes off
    assert states[-1] == "AS_OFF"


def test_mock_vcu_scenarios_pass() -> None:
    from StateMachine.mock_vcu import SCENARIOS

    for scenario in SCENARIOS.values():
        scenario(lambda _msg: None)


def test_step_that_keeps_raising_still_trips_link_watchdog() -> None:
    """A packet whose step() raises does not count as a live link."""
    clock = FakeClock()
    node = StateMachineNode(clock=clock)
    node.process(make_ready_telemetry(timestamp=1.0))

    def broken_step(*args, **kwargs):
        raise RuntimeError("simulated state machine bug")

    node.state_machine.step = broken_step

    for i in range(1, 16):
        clock.now += DT
        try:
            node.process(make_ready_telemetry(timestamp=1.0 + i * DT))
        except RuntimeError:
            pass  # the pipeline base class logs and skips these
    status, _ = node.check_link()
    assert status.current_state == MacroState.AS_EMERGENCY
