"""Registry for pluggable Level 2 autonomous missions."""

from typing import Dict, Type, Optional
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PIPELINE_DIR = os.path.join(ROOT_DIR, 'Pipeline')
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from messages import Mission
from StateMachine.level2.base import MissionBehavior


class MissionRegistry:
    """Thread-safe catalog of mission implementations indexed by Mission enum."""

    _registry: Dict[Mission, Type[MissionBehavior]] = {}

    @classmethod
    def register(cls, mission: Mission, behavior_cls: Type[MissionBehavior]) -> None:
        """Register a mission implementation class.
        
        Args:
            mission: The Mission enum key.
            behavior_cls: Concrete subclass of MissionBehavior.
        """
        cls._registry[mission] = behavior_cls

    @classmethod
    def create(cls, mission: Mission) -> Optional[MissionBehavior]:
        """Instantiate a mission behavior for the given mission.
        
        Args:
            mission: The Mission enum key.
            
        Returns:
            An instantiated MissionBehavior, or None if not registered.
        """
        behavior_cls = cls._registry.get(mission)
        if behavior_cls is None:
            return None
        return behavior_cls()

    @classmethod
    def is_registered(cls, mission: Mission) -> bool:
        """Check if a handler is registered for the specified mission."""
        return mission in cls._registry

    @classmethod
    def clear(cls) -> None:
        """Clear all registered missions (useful in test harnesses)."""
        cls._registry.clear()
