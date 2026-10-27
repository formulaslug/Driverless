# ControlPlanning

## File responsibilities

| File | Function |
|---|---|
| `controller.py` | `Controller` class — orchestrates one control-loop update: runs longitudinal PID, converts to throttle/brake; runs pure pursuit, converts to steering command. |
| `longitudinal.py` | PID controller. `v, v_desired, t` -> desired acceleration `a_des`. 
| `lateral.py` | Pure pursuit path tracker. `waypoints, x, y, yaw, v` -> steering angle `delta` (radians, front-wheel). |
| `throttle_control.py` | Converts `a_des, v` → `throttle_output, brake_output`, not used in current implementation |
| `steering_control.py` | Converts `delta` (radians) -> `steering_wheel_angle` (degrees) via `STEERING_RATIO`, clipped to `max_steering_angle`. |
| `config.py` | All tunable gains and vehicle physical parameters. |

## Inputs

**`Controller.__init__(K_p, K_i, K_d, K_dd, min_lookahead, wheelbase)`**
- `K_p, K_i, K_d` — longitudinal PID gains (floats)
- `K_dd` — pure pursuit lookahead scaler (float)
- `min_lookahead` — minimum lookahead distance, meters (float)
- `wheelbase` — vehicle wheelbase, meters (float)

**`Controller.update(x, y, yaw, v, t, v_desired, waypoints)`**
- `x, y` — current vehicle position, meters (float)
- `yaw` — current heading, radians (float)
- `v` — current velocity, m/s (float)
- `t` — current timestamp, seconds (float)
- `v_desired` — target velocity, m/s (float, scalar — not curvature-adjusted)
- `waypoints` — ordered list of `(x, y)` points defining the path

None of these inputs are validated, type-checked, or checked for staleness/finiteness before use.

## Outputs

**`Controller.update(...)` returns:**
- `throttle_output` — float in [0, 1]
- `brake_output` — float in [0, 1] (mutually exclusive with throttle — never both nonzero)
- `steer_output` — float, degrees, clipped to ±`max_steering_angle` (hardcoded to 30° in `controller.py`)

There is no metadata (timestamp, validity flag, saturation flag) attached to the output — just the three raw values.

## External calls

- **numpy** — used in `lateral.py` (`sqrt`, `sin`, `arctan2`, `arctan`), `throttle_control.py` and `steering_control.py` (`clip`).
- **`config.py`** — imported directly by `throttle_control.py` (vehicle physical constants) and `steering_control.py` (`STEERING_RATIO`). `longitudinal.py` and `lateral.py` receive gains as constructor arguments rather than importing config directly.
- No filesystem, network, or process calls anywhere in the control core.
- No external clock/logging adapter — `t` is passed in by the caller; there is no internal logging or diagnostics output.

## Failure modes (current behavior — not yet hardened)

| Condition | Current behavior |
|---|---|
| `t` non-increasing or equal to previous `t` (`dt <= 0`) | Division by `dt` in the derivative term will raise `ZeroDivisionError` or produce a divide-by-zero/inf if `dt = 0`; negative `dt` silently produces a nonsensical derivative. Not checked. |
| Empty `waypoints` list | `lateral.py` indexes `waypoints[-1]` and `waypoints[closest_idx:]` — will raise `IndexError`. |
| NaN or infinite input (`x, y, yaw, v, v_desired`) | Not checked anywhere; propagates through the PID/pure-pursuit math and will produce NaN/inf actuator commands, which are **not** caught by the `np.clip` calls (`np.clip` passes NaN through unchanged). |
| Integral windup | Mitigated — `net_integral` is clamped to ±10.0 in `longitudinal.py`. Currently inert in practice since `K_i = 0.00` in config. |
| Throttle/brake saturation | Handled — `accel_to_throttle_brake` clips both outputs to [0, 1]. No saturation flag is raised or reported. |
| Steering saturation | Handled — `delta_to_actuator` clips to ±`max_steering_angle`. No saturation flag is raised or reported. |
| Controller numerical failure (exception anywhere in the update chain) | Not caught. An exception in any submodule will propagate up through `Controller.update` uncaught — there is no safe/fallback command issued. |
| Stale vehicle state or watchdog/emergency-stop condition | Not implemented — there is no concept of state freshness, health monitoring, or emergency override in this code. |
| Config missing or out of range | Not validated at load — `config.py` is imported as-is; a missing or absurd value (e.g. `K_i` negative, `wheelbase = 0`) is not caught before use. |
| Vehicle physical constants unset | `M`, `R_WHEEL`, `J_M`, `J_WHEEL` are placeholder values (1.0) in `config.py`. Torque/throttle output computed from these is not physically meaningful until real values are substituted. |

## Known gaps ahead of vehicle deployment

- No simulation or unit test coverage yet.
- No curvature-based speed profiling — `v_desired` is a flat scalar, not reduced for upcoming corners.
- No input validation, exception handling, or fallback/safe-command behavior.
- No diagnostics/telemetry output (tracking error, saturation state, timing).
- No emergency-stop or watchdog integration.
- Real vehicle physical parameters not yet measured.