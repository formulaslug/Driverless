"""Constants and configuration parameters for the FSAE Driverless State Machine."""

# Operational timing and frequencies
STATE_MACHINE_TICK_RATE_HZ: float = 50.0
STATE_MACHINE_DEFAULT_DT_S: float = 1.0 / STATE_MACHINE_TICK_RATE_HZ

# AS_Ready lockout timer (PDR specification: >= 5.0 seconds)
AS_READY_LOCKOUT_DURATION_S: float = 5.0

# Standstill detection parameters for AS_Driving -> AS_Finished
STANDSTILL_VELOCITY_THRESHOLD_MPS: float = 0.05  # |v_x| < 0.05 m/s
STANDSTILL_MIN_DURATION_S: float = 0.5           # continuous duration required at standstill
STANDSTILL_BUFFER_CAPACITY: int = 100            # ring buffer capacity (2.0s at 50 Hz)

# Watchdog timeout limits
VCU_WATCHDOG_TIMEOUT_S: float = 0.2              # 200 ms timeout per FSAE rules
LEVEL2_WATCHDOG_TIMEOUT_S: float = 0.2           # 200 ms timeout for mission planner

# Default actuation commands
BRAKE_SERVICE_NORM: float = 1.0
BRAKE_EBS_NORM: float = 1.0
BRAKE_ZERO_NORM: float = 0.0
THROTTLE_ZERO_NORM: float = 0.0
STEERING_ZERO_RAD: float = 0.0

# Rolling counter rollover limit
HEARTBEAT_MAX: int = 0xFFFFFFFF
