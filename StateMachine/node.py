"""Pipeline Node integration for the FSAE Driverless State Machine."""

from typing import Optional, List, Union
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PIPELINE_DIR = os.path.join(ROOT_DIR, 'Pipeline')
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from node import Node
from messages import (
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
            self._latest_external_actuation = message
            return None

        if isinstance(message, VcuTelemetryMessage):
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
