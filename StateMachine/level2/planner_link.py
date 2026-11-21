from Pipeline.messages import (
    DiagnosticErrorCode,
    ActuationCommandMessage,
    Level2StatusMessage,
)
from StateMachine.level2.base import SubStateResult
from StateMachine.constants import (
    LEVEL2_WATCHDOG_TIMEOUT_S
)

from typing import Optional, Union

class PlannerLink:
    """Message processor that filters and caches messages.
    
    Each ActuationCommandMessage and Level2StatusMessage is passed into
    PlannerLink. Each message is thrown out if older than the current cached
    message, and a DiagnosticErrorCode is returned if a message is not
    received for an extended duration. SubStateResults are constructed from
    cached messages.
    
    Typical Usage Example:
    
        link = PlannerLink()
        state_machine = Level1StateMachine(planner_link=link)
        node = StateMachineNode(state_machine)
        
    Inputs:
        ActuationCommandMessage: Cached from external planner.
        Level2StatusMessage: Cached from external planner.
        
    Outputs:
        A SubStateResult or DiagnosticErrorCode per tick. SubStateResult is
        built from cached messages.
    
    Args:
        timeout_s: Length of timeout
    """
    def __init__(self, timeout_s: float = LEVEL2_WATCHDOG_TIMEOUT_S):
        self._timeout_s: float = timeout_s
        self._run_start_time: Optional[float] = None
        self._latest_cmd: Optional[ActuationCommandMessage] = None
        self._latest_status: Optional[Level2StatusMessage] = None
        
    def receive(self, message: Union[ActuationCommandMessage, Level2StatusMessage]) -> None:
        """Receives a message from the StateMachineNode, checks it, and caches it.
        
        Receive checks if the received message is older than the currently
        cached message. If so, it is discarded. If not, it is cached.
        
        Args:
            message: Message to be checked and cached.
        """
        if isinstance(message, Level2StatusMessage):
            # Filter older messages than current
            if (self._latest_status is not None
                and message.timestamp < self._latest_status.timestamp):
                return None
            self._latest_status = message
        else:
            if (self._latest_cmd is not None
                and message.timestamp < self._latest_cmd.timestamp):
                return None
            self._latest_cmd = message
                
    def start(self, now: float) -> None:
        """Initializes start time.
        
        Args:
            now: Timestamp to be stored.
        """
        self._run_start_time = now
        
    def tick(self, now: float) -> SubStateResult | DiagnosticErrorCode:
        """Builds and returns a SubStateResult based on current cached messages.
        
        Returns a DiagnosticErrorCode if the latest command hasn't arrived or
        if it is too stale.
        
        Args:
            now: Timestamp when tick is called.
        Returns:
            A SubStateResult derived from the current cached messages. If there
            is no command cached, or it is stale, a WATCHDOG_TIMEOUT_LEVEL2
            is returned instead.
        """
        # If start has not been run, or run after stop, return error code.
        if self._run_start_time is None:
            return DiagnosticErrorCode.WATCHDOG_TIMEOUT_LEVEL2
        
        # If no command has arrived or is too stale, return error code.
        if self._latest_cmd is None or now - self._latest_cmd.timestamp > self._timeout_s:
            return DiagnosticErrorCode.WATCHDOG_TIMEOUT_LEVEL2
        
        # If no latest status, or latest status is from previous run, only return actuation command.
        if self._latest_status is None or self._latest_status.timestamp < self._run_start_time:
            return SubStateResult(actuation_command=self._latest_cmd)
        
        return SubStateResult(next_substate=self._latest_status.active_substate,
                              actuation_command=self._latest_cmd,
                              emergency_stop=self._latest_status.emergency_stop,
                              mission_finished=self._latest_status.mission_finished)
    
    def stop(self) -> None:
        """Resets internal run start time variable to None."""
        self._run_start_time = None
            