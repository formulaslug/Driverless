"""Pipeline Node integration for the FSAE Driverless State Machine."""

import threading
import time
from typing import Callable, List, Optional, Union

from Pipeline.node import Node
from Pipeline.messages import (
    ActuationCommandMessage,
    Level2StatusMessage,
    VcuTelemetryMessage,
)
from StateMachine.constants import STATE_MACHINE_DEFAULT_DT_S, VCU_WATCHDOG_TIMEOUT_S
from StateMachine.level1_state_machine import Level1StateMachine


class StateMachineNode(Node):
    """Pipeline node that runs the Level 1 state machine.

    Each VcuTelemetryMessage runs one state machine cycle. Actuation commands
    and Level 2 status messages from an external planner are cached, and the
    state machine decides on each cycle whether they are fresh enough to use.
    A watchdog thread also emits outputs if telemetry stops arriving entirely,
    because process() is never called then and no packet-gap check can run.

    Typical usage example:
        state_machine = Level1Statemachine(planner_link=PlannerLink())
        node = StateMachineNode(state_machine=state_machine)
        node.start()

    Inputs:
        VcuTelemetryMessage: Runs one state machine cycle.
        ActuationCommandMessage: Cached by state machine planner link.
        Level2StatusMessage: Cached by state machine planner link.

    Outputs:
        A [StateMachineStatusMessage, ActuationCommandMessage] pair per cycle.
        The actuation command is gated by the state machine and always safe to
        send to the VCU.

    Args:
        state_machine: Level1StateMachine to drive.
        name: Node name, also used to name the watchdog thread.
        queueSize: Maximum number of queued input messages.
        clock: Monotonic clock in seconds used to time the VCU link. Tests can
            inject a fake clock.

    Attributes:
        state_machine: The Level1StateMachine driven by this node.
    """

    def __init__(
        self,
        state_machine: Level1StateMachine,
        name: str = "state_machine",
        queueSize: int = 16,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(name, queueSize=queueSize)
        self.state_machine = state_machine
        self._clock = clock
        # One lock guards the state machine and everything below: the pipeline
        # thread and the watchdog thread both drive the state machine.
        self._lock: threading.Lock = threading.Lock()
        self._last_rx_clock: Optional[float] = None
        self._last_rx_timestamp: Optional[float] = None
        self._watchdog_stop = threading.Event()
        self._watchdog = threading.Thread(target=self._watchdog_loop, name=f"{name}-watchdog", daemon=True)

    def process(self, message: Union[VcuTelemetryMessage,
                                     ActuationCommandMessage,
                                     Level2StatusMessage]) -> Optional[List]:
        """Handles one incoming pipeline message.

        Planner messages are only cached. A telemetry packet runs one state
        machine cycle and counts as a live VCU link, but only after step()
        returns.

        Args:
            message: Incoming message. ActuationCommandMessage and
                Level2StatusMessage come from an external planner and are
                cached for the next cycle. VcuTelemetryMessage runs a cycle.

        Returns:
            A [StateMachineStatusMessage, ActuationCommandMessage] pair for a
            telemetry message, or None for a cached or ignored message.
        """
        with self._lock:
            if isinstance(message, (ActuationCommandMessage, Level2StatusMessage)):
                self.state_machine.receive_planner_message(message)
                return None
            if isinstance(message, VcuTelemetryMessage):
                status, actuation = self.state_machine.step(message)
                # Only a packet that produced outputs counts as a live link, so a
                # step() that keeps raising still ends in a watchdog trip.
                self._last_rx_clock = self._clock()
                self._last_rx_timestamp = message.timestamp
                return [status, actuation]
        return None

    def check_link(self) -> Optional[List]:
        """Trips the VCU watchdog if telemetry has been silent too long.

        Called periodically by the watchdog thread. Does nothing until the
        first telemetry packet arrives.

        Returns:
            The (status, actuation) outputs to emit if the watchdog tripped,
            otherwise None.
        """
        with self._lock:
            if self._last_rx_clock is None:
                return None
            silent_for = self._clock() - self._last_rx_clock
            if silent_for <= VCU_WATCHDOG_TIMEOUT_S:
                return None
            # Express the trip time on the telemetry clock so logs line up with packets.
            outputs = self.state_machine.handle_telemetry_timeout(self._last_rx_timestamp + silent_for)
        return list(outputs) if outputs is not None else None

    def _watchdog_loop(self) -> None:
        """Runs check_link() every tick and emits any outputs until shutdown."""
        while not self._watchdog_stop.wait(STATE_MACHINE_DEFAULT_DT_S):
            self._emitResult(self.check_link())

    def start(self) -> None:
        """Starts the node and its VCU watchdog thread."""
        super().start()
        self._watchdog.start()

    def onShutdown(self):
        """Stops the watchdog thread and waits for it to exit.

        Returns:
            None, so no final outputs are emitted.
        """
        self._watchdog_stop.set()
        if self._watchdog.is_alive():
            self._watchdog.join()
        return None
