from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Optional, List
import numpy as np

# ---------------------------------------------------------------------------
# State Machine & VCU Enums
# ---------------------------------------------------------------------------

class MacroState(IntEnum):
    """FSAE Driverless Level 1 Macro-States."""
    AS_OFF = 0
    AS_READY = 1
    AS_DRIVING = 2
    AS_FINISHED = 3
    AS_EMERGENCY = 4
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
    """Autonomous Mission Indicator (AMI) mission selections."""
    NOT_SELECTED = 0
    MANUAL = 1
    INSPECTION = 2
    ACCELERATION = 3
    SKIDPAD = 4
    AUTOCROSS = 5
    TRACKDRIVE = 6


class DiagnosticErrorCode(IntFlag):
    """Structured diagnostic fault bitmask flags for AS_Emergency."""
    NONE = 0
    SDC_OPEN = 1 << 0
    EBS_ACTIVATED_UNEXPECTED = 1 << 1
    WATCHDOG_TIMEOUT_VCU = 1 << 2
    WATCHDOG_TIMEOUT_LEVEL2 = 1 << 3
    LEVEL2_EXCEPTION = 1 << 4
    DBS_CHECK_FAILED = 1 << 5
    BRAKE_CHECK_FAILED = 1 << 6
    TS_INACTIVE = 1 << 7
    DSMS_UNEXPECTED_DROP = 1 << 8
    STANDSTILL_TIMEOUT = 1 << 9
    ACTUATION_LIMIT_EXCEEDED = 1 << 10
    PRECONDITION_TIMEOUT = 1 << 11


# ---------------------------------------------------------------------------
# State Machine & Vehicle Control Messages
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class VcuTelemetryMessage:
    """Inbound telemetry packet from the Vehicle Control Unit (VCU) bridge.

    Every safety input is required: a bridge that forgets a field must fail
    loudly, never silently report "TS active, brakes engaged, SDC closed".
    """
    timestamp: float
    ebs_status: EbsStatus
    ebs_sound_active: bool
    dsms_status: bool
    dbs_check_ok: bool
    ts_status: bool
    ami_mission: Mission
    brake_status: bool
    sdc_status: bool
    res_go_signal: bool
    velocity: np.ndarray

    def __post_init__(self) -> None:
        self.velocity = as_float_array(self.velocity)


def as_float_array(value) -> np.ndarray:
    """Canonical array form for numeric message fields: float64, at least 1-D."""
    return np.atleast_1d(np.asarray(value, dtype=np.float64))


@dataclass
class Level2StatusMessage:
    """Internal Level 2 behavioral sub-state and execution status."""
    timestamp: float
    active_substate: str
    mission_finished: bool
    emergency_stop: bool
    fault_flags: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=bool))
    progress: float = 0.0


@dataclass
class ActuationCommandMessage:
    """Outbound Level 3 low-level actuation commands."""
    timestamp: float
    steering: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    throttle: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    braking: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float64))
    gated: bool = False

    def __post_init__(self) -> None:
        # Accept floats, 0-D arrays or lists, but always store float64 arrays of ndim >= 1.
        self.steering = as_float_array(self.steering)
        self.throttle = as_float_array(self.throttle)
        self.braking = as_float_array(self.braking)


@dataclass
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
