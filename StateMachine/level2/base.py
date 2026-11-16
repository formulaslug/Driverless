"""Abstract base classes and data transfer contracts for Level 2 behavioral sub-states."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from Pipeline.messages import VcuTelemetryMessage, ActuationCommandMessage


@dataclass(frozen=True, slots=True)
class MissionContext:
    """Execution context passed to Level 2 sub-states every tick."""
    telemetry: VcuTelemetryMessage
    dt: float
    mission_time: float
    current_substate: str


@dataclass(frozen=True, slots=True)
class SubStateResult:
    """Output emitted by a sub-state update."""
    next_substate: Optional[str] = None
    mission_finished: bool = False
    emergency_stop: bool = False
    actuation_command: Optional[ActuationCommandMessage] = None


class MissionBehavior(ABC):
    """Level 2 plug-in interface (the PDR's BehavioralSubState concept).

    Level 1 calls on_enter once on RES Go, update every DS_Driving tick, and
    on_exit when DS_Driving ends (interrupted=True for DS_Emergency). Sub-states
    are the mission's own business; report the current one via next_substate.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the mission."""
        pass

    @abstractmethod
    def on_enter(self, context: MissionContext) -> None:
        """Called when Level 1 transitions to DS_Driving for this mission."""
        pass

    @abstractmethod
    def update(self, context: MissionContext) -> SubStateResult:
        """Tick update called every cycle while in DS_Driving."""
        pass

    @abstractmethod
    def on_exit(self, context: MissionContext, interrupted: bool = False) -> None:
        """Called when DS_Driving exits (either finished, emergency, or shutdown)."""
        pass
