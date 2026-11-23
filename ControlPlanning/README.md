# Control Planning

Turns a planned path and a target speed into the two commands the vehicle needs: **throttle** and **steering wheel angle**.

## Usage

```python
from controller import Controller

controller = Controller()
throttle, steering_wheel_deg = controller.update(x, y, yaw, v, t, v_desired, waypoints)
```

## Input 

| Parameter | Type | Description |
|---|--|---|
| `x` | float | Vehicle x position from SLAM|
| `y` | float | Vehicle y position from SLAM|
| `yaw` | float | Vehicle heading (radians) from SLAM|
| `v` | float | Current speed (m/s) from IMU|
| `t` | float | Current time (s). from IMU|
| `v_desired` | float | Target speed (m/s) from PathPlanning|
| `waypoints` | np.ndarray | Smoothed waypoints `[x, y]` from PathPlanning|

## Output (to VCU)

| Field | Type | Description |
|---|---|---|
| `throttle` | float | Throttle in `[-0.2, 1.0]`. Negative values are regen braking |
| `steering_wheel_deg` | float | Desired steering wheel angle in degrees |

## Layout

| Path | Description |
|---|---|
| `controller.py` | Combines lateral, longitudinal and the output mappings |
| `lateral/lateral` | Holds chosen lateral algorithms |
| `lateral/pure_pursuit` | Pure Pursuit algorithm to get delta |
| `longitudinal/pid.py` | PID controller to get desired acceleration |
| `longitudinal/gain_scheduler.py` | Holds the gain profiles and ramps linearly between them |
| `longitudinal/longitudinal.py` | Holds longitduinal controller(PID)|
| `throttle_control.py` | converts acceleration to throttle percent |
| `steering_control.py` | converts delta to steering wheel angle |
| `control_config.py` | All vehicle and limit constants |
| `tune.py` | Tunes the pid controller using  the Full Vehicle Sim |
| `testing/` | Tests |

## Requirements
- numpy
- matplotlib
- SimsData FullVehichleSim libraries (only for tuning)