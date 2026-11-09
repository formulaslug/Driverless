"""Abstract base classes and data transfer contracts for Level 2 behavioral sub-states."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PIPELINE_DIR = os.path.join(ROOT_DIR, 'Pipeline')
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from messages import VcuTelemetryMessage, ActuationCommandMessage


@dataclass
class MissionContext:
    """Execution context passed to Level 2 sub-states every tick."""
    telemetry: VcuTelemetryMessage
    dt: float
    mission_time: float
    current_substate: str
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubStateResult:
    """Output emitted by a sub-state update."""
    next_substate: Optional[str] = None
    mission_finished: bool = False
    emergency_stop: bool = False
    actuation_command: Optional[ActuationCommandMessage] = None
    diagnostic_code: int = 0


class BehavioralSubState(ABC):
    """Abstract base class for modular Level 2 driving behaviors."""

    @abstractmethod
    def on_enter(self, context: MissionContext) -> None:
        """Called once upon entering this sub-state.
        
        Args:
            context: Current mission execution context.
        """
        pass

    @abstractmethod
    def update(self, context: MissionContext) -> SubStateResult:
        """Executed periodically (e.g. 50 Hz) while this sub-state is active.
        
        Args:
            context: Current mission execution context.
            
        Returns:
            SubStateResult specifying commands, next sub-state, or completion flags.
        """
        pass

    @abstractmethod
    def on_exit(self, context: MissionContext, interrupted: bool = False) -> None:
        """Cleanup hook called prior to transitioning out of this sub-state.
        
        Args:
            context: Current mission execution context.
            interrupted: True if exited abnormally (e.g. emergency stop or fault).
        """
        pass


class MissionBehavior(ABC):
    """Abstract interface for a full autonomous mission composed of sub-states."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the mission."""
        pass

    @abstractmethod
    def on_enter(self, context: MissionContext) -> None:
        """Called when Level 1 transitions to AS_Driving for this mission."""
        pass

    @abstractmethod
    def update(self, context: MissionContext) -> SubStateResult:
        """Tick update called every cycle while in AS_Driving."""
        pass

    @abstractmethod
    def on_exit(self, context: MissionContext, interrupted: bool = False) -> None:
        """Called when AS_Driving exits (either finished, emergency, or shutdown)."""
        pass
