"""Pipeline Node integration for the FSAE Driverless State Machine."""

from typing import Optional, List, Union
import threading

from Pipeline.node import Node
from Pipeline.messages import (
    VcuTelemetryMessage,
    ActuationCommandMessage,
    StateMachineStatusMessage,
    MacroState,
)
from StateMachine.constants import LEVEL2_WATCHDOG_TIMEOUT_S
from StateMachine.level1_state_machine import Level1StateMachine


class StateMachineNode(Node):
    """Pipeline node executing the FSAE Driverless Hierarchical State Machine.
    
    Subscribes to VcuTelemetryMessage (from VCU or MockVCU bridge) and optionally
    to ActuationCommandMessage (from Level 2 / Controls path planner).
    Emits [StateMachineStatusMessage, ActuationCommandMessage].
    """

    def __init__(
        self,
        name: str = "state_machine",
        queueSize: int = 16,
        state_machine: Optional[Level1StateMachine] = None,
    ) -> None:
        super().__init__(name, queueSize=queueSize)
        self.state_machine: Level1StateMachine = state_machine or Level1StateMachine()
        self._latest_external_actuation: Optional[ActuationCommandMessage] = None
        self._lock: threading.Lock = threading.Lock()

    def process(
        self, message: Union[VcuTelemetryMessage, ActuationCommandMessage]
    ) -> Optional[List[Union[StateMachineStatusMessage, ActuationCommandMessage]]]:
        """Process incoming telemetry or external actuation requests.
        
        Args:
            message: VcuTelemetryMessage or ActuationCommandMessage.
            
        Returns:
            List of [StateMachineStatusMessage, gated ActuationCommandMessage] on telemetry,
            or None on external actuation caching.
        """
        if isinstance(message, ActuationCommandMessage):
            with self._lock:
                self._latest_external_actuation = message
            return None

        if isinstance(message, VcuTelemetryMessage):
            with self._lock:
                actuation_to_use = self._latest_external_actuation
                if (
                    actuation_to_use is not None
                    and (message.timestamp - actuation_to_use.timestamp) > LEVEL2_WATCHDOG_TIMEOUT_S
                ):
                    # Command is stale (> 200ms); invalidate cached command
                    actuation_to_use = None
                    self._latest_external_actuation = None

            status_msg, gated_actuation = self.state_machine.step(
                telemetry=message,
                external_actuation=actuation_to_use,
            )
            return [status_msg, gated_actuation]

        return None

    def request_reset(self, telemetry: VcuTelemetryMessage) -> bool:
        """Thread-safe request to reset emergency state."""
        with self._lock:
            return self.state_machine.request_reset(telemetry)
