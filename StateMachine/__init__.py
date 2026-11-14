"""FSAE Driverless Hierarchical State Machine (HSM)."""

from Pipeline.messages import DSSIStatus, MacroState, Mission, DiagnosticErrorCode
from StateMachine.constants import (
    STATE_MACHINE_TICK_RATE_HZ,
    STATE_MACHINE_DEFAULT_DT_S,
    AS_READY_LOCKOUT_DURATION_S,
    STANDSTILL_VELOCITY_THRESHOLD_MPS,
    STANDSTILL_MIN_DURATION_S,
    STANDSTILL_BUFFER_CAPACITY,
    FINISH_STANDSTILL_TIMEOUT_S,
    VCU_WATCHDOG_TIMEOUT_S,
    LEVEL2_WATCHDOG_TIMEOUT_S,
)
from StateMachine.standstill_detector import StandstillDetector
from StateMachine.gatekeeper import gate_command
from StateMachine.level1_state_machine import (
    Level1StateMachine,
    unpack_diagnostic_flags,
    VALID_DRIVERLESS_MISSIONS,
)
from StateMachine.level2 import (
    MissionBehavior,
    MissionContext,
    SubStateResult,
    MissionRegistry,
    InspectionMission,
)

__all__ = [
    "DSSIStatus",
    "MacroState",
    "Mission",
    "DiagnosticErrorCode",
    "STATE_MACHINE_TICK_RATE_HZ",
    "STATE_MACHINE_DEFAULT_DT_S",
    "AS_READY_LOCKOUT_DURATION_S",
    "STANDSTILL_VELOCITY_THRESHOLD_MPS",
    "STANDSTILL_MIN_DURATION_S",
    "STANDSTILL_BUFFER_CAPACITY",
    "FINISH_STANDSTILL_TIMEOUT_S",
    "VCU_WATCHDOG_TIMEOUT_S",
    "LEVEL2_WATCHDOG_TIMEOUT_S",
    "StandstillDetector",
    "gate_command",
    "Level1StateMachine",
    "unpack_diagnostic_flags",
    "VALID_DRIVERLESS_MISSIONS",
    "MissionBehavior",
    "MissionContext",
    "SubStateResult",
    "MissionRegistry",
    "InspectionMission",
]
