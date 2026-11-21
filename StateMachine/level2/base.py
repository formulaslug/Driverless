"""SubStateResult Dataclass"""

from dataclasses import dataclass
from typing import Optional

from Pipeline.messages import ActuationCommandMessage

@dataclass(frozen=True, slots=True)
class SubStateResult:
    """Output emitted by a sub-state update."""
    next_substate: Optional[str] = None
    mission_finished: bool = False
    emergency_stop: bool = False
    actuation_command: Optional[ActuationCommandMessage] = None
