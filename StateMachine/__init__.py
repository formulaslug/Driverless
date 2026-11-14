"""FSAE Driverless Hierarchical State Machine (HSM)."""

from Pipeline.messages import DSSIStatus, MacroState, Mission, DiagnosticErrorCode
from StateMachine.constants import (
    STATE_MACHINE_TICK_RATE_HZ,
    STATE_MACHINE_DEFAULT_DT_S,
    AS_READY_LOCKOUT_DURATION_S,
    STANDSTILL_VELOCITY_THRESHOLD_MPS,
    STANDSTILL_MIN_DURATION_S,
    STANDSTILL_BUFFER_CAPACITY,
    VCU_WATCHDOG_TIMEOUT_S,
    LEVEL2_WATCHDOG_TIMEOUT_S,
)
from StateMachine.standstill_detector import StandstillDetector  # noqa: E402
from StateMachine.gatekeeper import gate_command  # noqa: E402
from StateMachine.level1_state_machine import (  # noqa: E402
    Level1StateMachine,
    unpack_diagnostic_flags,
    VALID_DRIVERLESS_MISSIONS,
)
from StateMachine.level2 import (  # noqa: E402
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
