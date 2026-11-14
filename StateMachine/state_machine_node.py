"""Pipeline Node integration for the FSAE Driverless State Machine."""

import threading
import time
from typing import Callable, List, Optional

from Pipeline.node import Node
from Pipeline.messages import (
    ActuationCommandMessage,
    Level2StatusMessage,
    VcuTelemetryMessage,
)
from StateMachine.constants import STATE_MACHINE_DEFAULT_DT_S, VCU_WATCHDOG_TIMEOUT_S
from StateMachine.level1_state_machine import Level1StateMachine


class StateMachineNode(Node):
    """Pipeline node running the Level 1 state machine.

    Inputs:  VcuTelemetryMessage (each one runs a state machine cycle),
             ActuationCommandMessage and Level2StatusMessage from an external planner
             (cached; the state machine decides whether they are fresh enough).
    Outputs: [StateMachineStatusMessage, gated ActuationCommandMessage] per cycle.

    A watchdog thread also emits those outputs if telemetry stops arriving
    entirely, because then process() is never called and no packet-gap check can run.
    """

    def __init__(
        self,
        name: str = "state_machine",
        queueSize: int = 16,
        state_machine: Optional[Level1StateMachine] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(name, queueSize=queueSize)
        self.state_machine = state_machine or Level1StateMachine()
        self._clock = clock
        # One lock guards the state machine and everything below: the pipeline
        # thread and the watchdog thread both drive the state machine.
        self._lock = threading.Lock()
        self._latest_actuation: Optional[ActuationCommandMessage] = None
        self._latest_level2_status: Optional[Level2StatusMessage] = None
        self._last_rx_clock: Optional[float] = None
        self._last_rx_timestamp: Optional[float] = None
        self._watchdog_stop = threading.Event()
        self._watchdog = threading.Thread(target=self._watchdog_loop, name=f"{name}-watchdog", daemon=True)

    def process(self, message) -> Optional[List]:
        with self._lock:
            if isinstance(message, ActuationCommandMessage):
                self._latest_actuation = message
                return None
            if isinstance(message, Level2StatusMessage):
                self._latest_level2_status = message
                return None
            if isinstance(message, VcuTelemetryMessage):
                self._last_rx_clock = self._clock()
                self._last_rx_timestamp = message.timestamp
                status, actuation = self.state_machine.step(
                    message,
                    external_actuation=self._latest_actuation,
                    external_status=self._latest_level2_status,
                )
                return [status, actuation]
        return None

    def check_link(self) -> Optional[List]:
        """Trip the VCU watchdog if no telemetry arrived for VCU_WATCHDOG_TIMEOUT_S.

        Returns the outputs to emit, or None. Called by the watchdog thread every tick.
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
        while not self._watchdog_stop.wait(STATE_MACHINE_DEFAULT_DT_S):
            self._emitResult(self.check_link())

    def start(self) -> None:
        super().start()
        self._watchdog.start()

    def onShutdown(self):
        self._watchdog_stop.set()
        if self._watchdog.is_alive():
            self._watchdog.join()
        return None
