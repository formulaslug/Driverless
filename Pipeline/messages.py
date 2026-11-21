from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Optional, List
import numpy as np

# ---------------------------------------------------------------------------
# State Machine & VCU Enums
# ---------------------------------------------------------------------------

class MacroState(IntEnum):
    """FSAE Driverless Level 1 Macro-States."""
    DS_OFF = 0
    DS_READY = 1
    DS_DRIVING = 2
    DS_FINISHED = 3
    DS_EMERGENCY = 4
    MANUAL_DRIVING = 5


class DSSIStatus(IntEnum):
    """Driverless System Status Indicator (DSSI) light states."""
    OFF = 0
    CONTINUOUS_YELLOW = 1
    FLASHING_YELLOW = 2
    FLASHING_BLUE = 3
    CONTINUOUS_BLUE = 4


class EbsStatus(IntEnum):
    """Emergency Brake System (EBS) hardware state."""
    UNAVAILABLE = 0
    ARMED = 1
    ACTIVATED = 2


class Mission(IntEnum):
    """Driverless Mission Indicator (DMI) mission selections."""
    NOT_SELECTED = 0
    MANUAL = 1
    INSPECTION = 2
    ACCELERATION = 3
    SKIDPAD = 4
    AUTOCROSS = 5
    TRACKDRIVE = 6


class DiagnosticErrorCode(IntFlag):
    """Fault one-hot bitmask reported while in DS_Emergency. Every flag is raised somewhere in Level 1.
    
    note: If you have more than 16 errors, 
          then change the _BIT_INDICES length in level1_state_machine.py
    """
    NONE                     = 0
    SDC_OPEN                 = 1 << 0
    EBS_ACTIVATED_UNEXPECTED = 1 << 1
    WATCHDOG_TIMEOUT_VCU     = 1 << 2
    WATCHDOG_TIMEOUT_LEVEL2  = 1 << 3
    LEVEL2_EXCEPTION         = 1 << 4  # reserved: was in-process mission exception; do not reuse.
    LEVEL2_EMERGENCY_STOP    = 1 << 5
    TS_INACTIVE              = 1 << 6
    DSMS_UNEXPECTED_DROP     = 1 << 7
    STANDSTILL_TIMEOUT       = 1 << 8


# ---------------------------------------------------------------------------
# State Machine & Vehicle Control Messages
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True, kw_only=True)
class VcuTelemetryMessage:
    """Inbound telemetry packet from the Vehicle Control Unit (VCU) bridge.

    Every safety input is required: a bridge that forgets a field must fail
    loudly, never silently report "TS active, brakes engaged, SDC closed".
    """
    timestamp: float
    ebs_status: EbsStatus
    ebs_sound_active: bool
    dsms_on: bool
    dbs_check_ok: bool
    ts_active: bool
    dmi_mission: Mission
    brakes_engaged: bool
    sdc_closed: bool
    res_go_signal: bool
    velocity: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "velocity", as_float_array(self.velocity))


def as_float_array(value) -> np.ndarray:
    """Canonical array form for numeric message fields: a read-only float64 copy, at least 1-D."""
    return as_readonly_array(value, np.float64)


def as_readonly_array(value, dtype) -> np.ndarray:
    """Private, read-only, at-least-1-D copy of ``value``.

    Frozen dataclasses only stop field reassignment; this also stops in-place
    edits like ``msg.steering[0] = 1.0``. Copying first means the caller's own
    array is never locked.
    """
    array = np.atleast_1d(np.array(value, dtype=dtype, copy=True))
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class Level2StatusMessage:
    """Internal Level 2 behavioral sub-state and execution status."""
    timestamp: float
    active_substate: str
    mission_finished: bool
    emergency_stop: bool
    fault_flags: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=bool))
    progress: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "fault_flags", as_readonly_array(self.fault_flags, bool))


@dataclass(frozen=True, slots=True)
class ActuationCommandMessage:
    """Outbound Level 3 low-level actuation commands."""
    timestamp: float
    steering: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    throttle: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    braking: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    gated: bool = False

    def __post_init__(self) -> None:
        # Accept floats, 0-D arrays or lists, but always store float64 arrays of ndim >= 1.
        object.__setattr__(self, "steering", as_float_array(self.steering))
        object.__setattr__(self, "throttle", as_float_array(self.throttle))
        object.__setattr__(self, "braking", as_float_array(self.braking))


@dataclass(frozen=True, slots=True)
class StateMachineStatusMessage:
    """Outbound Level 1 state machine telemetry, diagnostics, and gate status."""
    timestamp: float
    current_state: MacroState
    dssi_status: DSSIStatus
    error_code: int
    error_flags: np.ndarray
    heartbeat: int
    ebs_trigger_cmd: bool
    service_brake_cmd: bool
    active_mission: Mission
    active_substate: str
    ready_timer_elapsed: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "error_flags", as_readonly_array(self.error_flags, bool))


# ---------------------------------------------------------------------------
# Perception, Planning, and Odometry Messages
# ---------------------------------------------------------------------------

@dataclass
class FrameMessage:
    frameId: int
    frameName: str
    image: np.ndarray
    camera: object

@dataclass
class SegmentationMessage:
    frameId: int
    masks: Optional[np.ndarray]
    boxes: Optional[np.ndarray]
    classes: Optional[list]
    confidences: Optional[np.ndarray]
    numDetections: int

@dataclass
class ConeDistancesMessage:
    frameId: int
    coneDistances: list

@dataclass
class ConeMapMessage:
    frameId: int
    cones: list
    coneMap: list

@dataclass
class PathMessage:
    frameId: int
    smoothPath: Optional[np.ndarray]
    curvature: Optional[np.ndarray]

@dataclass
class OdometryMessage:
    frameId: int
    dx: float
    dy: float
    dyaw: float

# IMU input contract. ImuMotionEstimator accepts these once real sensor data is available.
@dataclass
class ImuSample:
    timestamp: float
    accel: np.ndarray
    gyro: np.ndarray
    mag: Optional[np.ndarray] = None

# Emitted by a node whose process() raised so join nodes and reorder buffers can
# discard the frame instead of waiting on it forever.
@dataclass
class FailedFrameMessage:
    frameId: int
    source: str
