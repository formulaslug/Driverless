# Driverless State Machine

This folder is the part of the car that decides whether the car is *allowed* to move, and filters every actuator command before it reaches the VCU. It is the FSAE Driverless state machine from the rulebook (`DS_Off`, `DS_Ready`, `DS_Driving`, `DS_Finished`, `DS_Emergency`), plus a plug-in slot where the autonomous team's driving logic lives.

If you want the *why* (rule references, the original design doc, the roadmap), read [StateMachinePDR.md](StateMachinePDR.md). This README is the *how*: how the pieces connect, what happens to a message as it moves through, and how to plug your own code in without breaking anything (you might have to scroll a bit). I'll try to word this as best as I can:

## Overview

There are three levels. Each one only talks to the level directly below it.

```
Level 1   Rulebook states        level1_state_machine.py
          "Is the car allowed to drive right now?"
              |
              |  only while in DS_Driving
              v
Level 2   Mission behavior       level2/   (or an external planner node)
          "What should the car do on this event?"
              |
              |  a requested ActuationCommandMessage
              v
Level 3   Actuation gate         gatekeeper.py
          "Is this command safe to send? If not, replace it."
              |
              v
             VCU
```

One of the biggest takeaways with this design is that **Level 2 never talks to the VCU directly.** Your planner simply *requests* a command. Level 1 decides whether it is driving time, and the gatekeeper decides what actually goes out. If you are not in `DS_Driving`, your command is thrown away and replaced with zero throttle, zero steering and the brake the current state demands. This way, you can write Level 2 code without worrying about the Level 1 rulebook for state machine safety.

Another important idea is that **the state machine only runs when VCU telemetry arrives.** There is no internal timer loop / expected heartbeat. One `VcuTelemetryMessage` in means one tick, and one tick means one `(StateMachineStatusMessage, ActuationCommandMessage)` pair out. The VCU is the clock, and this works in tandem with it.

## Where things live

| File | What it does |
| --- | --- |
| [state_machine_node.py](state_machine_node.py) | The Pipeline `Node` wrapper. Receives messages, runs a tick per telemetry packet, owns the "VCU went silent" watchdog thread. |
| [level1_state_machine.py](level1_state_machine.py) | Level 1. The rulebook states and every transition between them. Also decides who is driving in Level 2 and calls into it. |
| [level2/base.py](level2/base.py) | The Level 2 contract: `MissionBehavior` (the interface you implement), `MissionContext` (what you get each tick), `SubStateResult` (what you give back). |
| [level2/registry.py](level2/registry.py) | A lookup table from `Mission` enum value to the class that drives that mission. |
| [level2/missions/](level2/missions/) | Actual missions. `inspection.py` is the reference example. |
| [gatekeeper.py](gatekeeper.py) | Level 3. `gate_command()` clamps or replaces every outgoing command. |
| [standstill_detector.py](standstill_detector.py) | Decides "the car has actually stopped", used for `DS_Driving -> DS_Finished`. |
| [constants.py](constants.py) | Every timeout, limit and threshold. Change numbers here, not inline. |
| [mock_vcu.py](mock_vcu.py) | A fake VCU for testing, with scripted scenarios and an interactive shell. This shouldn't be on the actual car. |
| [tests/](tests/) | Tests for every flowchart arrow, the watchdogs, the gate and the pipeline wiring. |

The message types themselves (`VcuTelemetryMessage`, `ActuationCommandMessage`, `Level2StatusMessage`, `StateMachineStatusMessage`, and the `MacroState` / `Mission` / `DiagnosticErrorCode` enums) are defined in [Pipeline/messages.py](../Pipeline/messages.py), because other parts of the stack use them too.

## Following one packet from the top to the wheels

This is probably the easiest way to understand the flow.. you just have to do it. Let's say the car is sitting in `DS_Driving` running the Inspection mission, and the VCU sends a telemetry packet. Here is everything that happens before a steering command goes back out.

### 1. The node receives it

[`StateMachineNode.process()`](state_machine_node.py:73) gets every message sent to the node. It sorts them into two piles:

- `ActuationCommandMessage` and `Level2StatusMessage` (from an external planner, more on that later) are **just cached**. Nothing runs. The node returns `None`.
- `VcuTelemetryMessage` **runs one tick**: it calls `state_machine.step(telemetry, external_actuation=<cached>, external_status=<cached>)`.

After `step()` returns, the node records the wall clock time of this packet. That's what the watchdog thread watches (step 7).

### 2. Level 1 starts the tick

[`Level1StateMachine.step()`](level1_state_machine.py:197) does some bookkeeping first:

- `now` is the **telemetry timestamp**, not the computer's clock. Every time comparison in Level 1 and Level 2 uses this clock. Remember that, it matters a lot later.
- `dt` is the gap since the last packet.
- The velocity goes into the standstill detector.
- **Packet gap check:** if we're in `DS_Ready` or `DS_Driving` and `dt` is over 200 ms, the VCU link dropped long enough to count as lost, and we trip straight to `DS_Emergency`.

Then it dispatches on the current state with a `match`. Each state has its own `_step_ds_*` method that checks that state's exit conditions. We're in `DS_Driving`, so we go to `_step_ds_driving`.

### 3. `DS_Driving` checks hardware, then asks Level 2

[`_step_ds_driving()`](level1_state_machine.py:357) does this, in this order:

1. **Hardware faults first.** SDC open, EBS fired, DSMS dropped, TS off. Any of these trips `DS_Emergency` and Level 2 is never even called this tick.
2. **Ask Level 2 for a decision.** There are two possible sources, and exactly one is active per run:
   - If a `MissionBehavior` object exists (from the registry or injected), call [`_run_mission()`](level1_state_machine.py:488), which calls `behavior.update(context)`.
   - Otherwise, call [`_external_planner_result()`](level1_state_machine.py:506), which builds the same kind of answer from the cached planner messages.

   Either way, Level 1 ends up holding a `SubStateResult`. This is the trick that ties the whole design together: **Level 1 does not care where Level 2 lives.** An in-process mission and a separate planner node look identical by the time they reach this line.

### 4. Inside Level 2 (the Inspection mission)

`update()` gets a [`MissionContext`](level2/base.py) built by [`_context()`](level1_state_machine.py:536):

| Field | Meaning |
| --- | --- |
| `telemetry` | The full VCU packet for this tick (velocity, switches, etc.) |
| `dt` | Seconds since the previous tick |
| `mission_time` | Seconds since RES Go (so `0.0` at the start of the run) |
| `current_substate` | Whatever sub-state name you reported last tick |

The Inspection mission ([missions/inspection.py](level2/missions/inspection.py:64)) checks whether it's time to move to its next sub-state (`STEERING_SWEEP -> POWERTRAIN_RAMP -> CONTROLLED_STOP -> COMPLETE`), works out steering/throttle/brake for the sub-state it's in, and returns:

```python
SubStateResult(
    next_substate="STEERING_SWEEP",   # just a label, used for logs and status
    mission_finished=False,           # True = "I'm done, let me stop"
    emergency_stop=False,             # True = "something is wrong, trip EBS now"
    actuation_command=cmd,            # what I'd like the car to do
)
```

Notice what the mission *doesn't* do: it never changes the macro-state. It can only raise its hand (`mission_finished`, `emergency_stop`). Level 1 decides what that means.

If `update()` raises any exception, Level 1 catches it and trips `DS_Emergency` with `LEVEL2_EXCEPTION`. Your bug cannot crash the state machine, but it will stop the car.

### 5. Level 1 interprets the result

Back in `_step_ds_driving()`:

- `next_substate` is saved so it shows up in the status message.
- `emergency_stop=True` trips `DS_Emergency` (`LEVEL2_EMERGENCY_STOP`).
- `mission_finished=False` means "keep going", so the requested command is passed on.
- `mission_finished=True` starts the finish sequence. Level 1 keeps passing your command through (so you can keep braking) until the standstill detector says the car has been under 0.05 m/s for 0.5 s. Then it moves to `DS_Finished`. If the car is still rolling 5 s after you said you were finished, that's `STANDSTILL_TIMEOUT` and an emergency.

### 6. The gate and the outputs

Whatever happened, the tick ends in [`_outputs()`](level1_state_machine.py:618), which:

1. Bumps the **heartbeat** counter (wraps at 2^32).
2. Builds a `StateMachineStatusMessage`: current state, DSSI light color, latched fault bits, `ebs_trigger_cmd`, `service_brake_cmd`, mission, sub-state, DS_Ready timer.
3. Runs the requested command through [`gate_command()`](gatekeeper.py:75):
   - Not in `DS_Driving`, or no command: zero throttle, zero steering, and the state's brake (service brake in `DS_Ready`, full EBS brake in `DS_Finished` / `DS_Emergency`, released in `DS_Off` / manual). `gated=False`.
   - In `DS_Driving` with a command containing NaN/inf or mismatched array shapes: service-brake stop. `gated=False`.
   - In `DS_Driving` with a sane command: clamped to physical limits (steering to +/-0.4363 rad, throttle and brake to `[0, 1]`) and passed through. `gated=True`.

`gated=True` literally means "this came from Level 2 and was allowed through." Anything else was generated by the state machine itself.

The node returns `[status, actuation]`, and the Pipeline broadcasts both to every subscriber. That's the end of the tick.

### 7. Meanwhile, the watchdog thread

The packet gap check in step 2 only runs when a packet *arrives*. If the VCU goes completely silent, `process()` is never called and nothing would ever notice. So [`StateMachineNode`](state_machine_node.py:111) also runs a background thread that wakes every 20 ms and checks how long it's been since the last packet. Past 200 ms (in `DS_Ready` or `DS_Driving`), it calls `handle_telemetry_timeout()`, which trips the emergency and publishes one final status/actuation pair with EBS braking. Both threads share one lock, so they never step the state machine at the same time.

## Level 2, (base.py and registry.py)

This is where most people get lost, because the abstract classes don't tell you *when* anything gets called. Here's the lifecycle.

### When your mission object is created

Not at startup. I know, bear with me:

1. **`DS_Off -> DS_Ready`:** Level 1 reads the mission off the DMI (`telemetry.dmi_mission`) and **commits** to it. Changing the DMI selector after this point does nothing for the rest of the run.
2. **RES Go (`DS_Ready -> DS_Driving`):** [`_start_mission()`](level1_state_machine.py:459) asks for a behavior:
   - If you passed one to `Level1StateMachine(mission_behavior=...)`, it uses that (mostly for tests).
   - Otherwise it calls `MissionRegistry.create(mission)`, which looks up the class registered for that mission and constructs a **fresh instance with no arguments**.
   - If nothing is registered for that mission, there's no behavior, and Level 1 expects an **external planner** to drive instead (sub-state shows as `EXTERNAL_PLANNER`).
3. If there is a behavior, `on_enter(context)` is called once. `mission_time` is `0.0`.
4. **Every `DS_Driving` tick:** `update(context)` is called and must return a `SubStateResult`.
5. **When `DS_Driving` ends** (finished *or* emergency), `on_exit(context, interrupted)` is called once. `interrupted=True` means you're leaving because of an emergency. Then Level 1 drops the object.

### What the registry actually is

[`MissionRegistry`](level2/registry.py) is a dictionary from `Mission` enum to a class. That's all. The non-obvious part is *how things get into it*: each mission file calls `MissionRegistry.register(...)` at the bottom, at module level. So **a mission is only registered if its file gets imported.** The import chain that makes this happen is:

```
level1_state_machine.py
  imports StateMachine.level2.base
    which runs StateMachine/level2/__init__.py
      which imports StateMachine/level2/missions/__init__.py
        which imports inspection.py
          which calls MissionRegistry.register(Mission.INSPECTION, InspectionMission)
```

If you write a new mission and forget to import it in [missions/\_\_init\_\_.py](level2/missions/__init__.py), it silently isn't registered, Level 1 thinks an external planner is supposed to drive, and the car trips `WATCHDOG_TIMEOUT_LEVEL2` about 200 ms after RES Go.

### Two ways to be Level 2

| | In-process mission | External planner node |
| --- | --- | --- |
| What you write | A `MissionBehavior` subclass, registered for a `Mission` | A Pipeline `Node` that publishes `ActuationCommandMessage` (+ optionally `Level2StatusMessage`) to the state machine node |
| How it's called | Synchronously, inside the tick, every telemetry packet | Your own thread, at your own rate. Level 1 uses whatever you sent most recently. |
| What you can see | Only `MissionContext` (VCU telemetry + timing) | Anything in the pipeline: cone maps, paths, odometry, camera frames |
| Liveness check | None needed. A crash trips `LEVEL2_EXCEPTION`. | Your commands must be fresher than 200 ms or it trips `WATCHDOG_TIMEOUT_LEVEL2` |
| Good for | Open-loop or simple scripted missions (Inspection, a dumb accel test) | Anything that needs perception: Autocross, Trackdrive, Skidpad, real Acceleration |

**Pick one per mission.** If a mission is in the registry, the in-process class drives and any planner messages for that run are ignored. Right now only `INSPECTION` is registered, so `ACCELERATION`, `SKIDPAD`, `AUTOCROSS` and `TRACKDRIVE` all expect an external planner.

## Recipe: I'm the path planning node. How do I drive the car?

You're almost certainly an external planner, because you need cone data and `MissionContext` doesn't have any. Here's what you do.

**Wiring.** Your node sends commands into the state machine node, and listens to the state machine's status so you know what state the car is in and what time it is:

```python
planner.subscribe(sm_node)     # planner -> state machine (commands + status)
sm_node.subscribe(planner)     # state machine -> planner (status + gated actuation)
perception.subscribe(planner)  # whatever gives you cones
```

**The node.** A minimal sketch:

```python
import numpy as np
from Pipeline.node import Node
from Pipeline.messages import (
    ActuationCommandMessage, Level2StatusMessage, StateMachineStatusMessage,
    ConeMapMessage, MacroState,
)

class AccelerationPlanner(Node):
    def __init__(self):
        super().__init__("accel_planner", queueSize=4)
        self.cones = None

    def process(self, msg):
        if isinstance(msg, ConeMapMessage):
            self.cones = msg
            return None
        if not isinstance(msg, StateMachineStatusMessage):
            return None           # ignore the gated actuation echo

        t = msg.timestamp         # telemetry clock. Use THIS, not time.monotonic().

        if msg.current_state != MacroState.DS_DRIVING:
            # Not driving yet (or anymore). Reset, and still send a harmless
            # command so one is already cached the moment RES Go is pressed.
            self.reset()
            return [ActuationCommandMessage(timestamp=t, braking=1.0),
                    Level2StatusMessage(timestamp=t, active_substate="WAITING",
                                        mission_finished=False, emergency_stop=False)]

        steering, throttle, braking, substate, done = self.plan(t)
        return [
            ActuationCommandMessage(timestamp=t, steering=steering,
                                    throttle=throttle, braking=braking),
            Level2StatusMessage(timestamp=t, active_substate=substate,
                                mission_finished=done, emergency_stop=False),
        ]
```

The rules that matter:

1. **Stamp messages with the telemetry clock.** Level 1 checks freshness with `telemetry.timestamp - cmd.timestamp > 0.2`. If you use `time.time()` or `time.monotonic()`, those numbers are on a completely different clock and you'll either trip instantly or never trip. Copy the timestamp from the `StateMachineStatusMessage` you're responding to.
2. **Reply to every status, not only `DS_Driving` ones.** If your first command arrives *after* the first `DS_Driving` tick, Level 1 sees no command and trips `WATCHDOG_TIMEOUT_LEVEL2` immediately. Sending a brake-hold command while in `DS_Ready` is free, because the gate throws it away outside `DS_Driving`.
3. **Send a `Level2StatusMessage` if you ever want to finish.** It's the only way to say `mission_finished=True` (or `emergency_stop=True`). Without it the car drives until something else stops it. Status with a timestamp from before RES Go is ignored, so stale "finished" messages from a previous run can't end a new run early.
4. **Keep braking after you say you're finished.** Level 1 keeps passing your commands through until the car is at standstill. If you send zero brake and roll for 5 s, that's an emergency.
5. **You don't decide the final command.** Your steering gets clamped to +/-25 degrees (approximate rack limit, not verified yet) and NaN anywhere gets you a service-brake stop. Check `gated` on the actuation the state machine echoes back if you want to know whether you got through.
6. **Don't block the state machine.** The state machine node emits into your input queue with a blocking `put`. If your queue fills because you're slow, the state machine thread waits on you. Keep `process()` quick and drop stale inputs.

**Known gap:** the planner and the state machine subscribing to each other is a loop, and the Pipeline's end-of-stream shutdown waits for a `POISON` from *every* upstream. In a loop, each node waits for the other, so `graph.join()` can hang on shutdown. This doesn't affect driving, but it will need a fix in `Pipeline/` before a full-stack run shuts down cleanly.

## Recipe: adding a new in-process mission

Use this when the mission doesn't need perception, or when you want the logic to run inside the tick. Say you're adding a simple open-loop acceleration test.

**1. Make sure the mission exists in the enum.** `Mission` in [Pipeline/messages.py](../Pipeline/messages.py) already has `ACCELERATION`, `SKIDPAD`, `AUTOCROSS`, `TRACKDRIVE`. If you need a brand new one, add it there *and* to `VALID_DRIVERLESS_MISSIONS` in [level1_state_machine.py](level1_state_machine.py), and make sure the real DMI/VCU can actually select it.

**2. Write the class** in `level2/missions/acceleration.py`:

```python
from enum import Enum

from Pipeline.messages import Mission, ActuationCommandMessage
from StateMachine.level2.base import MissionBehavior, MissionContext, SubStateResult
from StateMachine.level2.registry import MissionRegistry


class AccelSubState(str, Enum):
    LAUNCH = "LAUNCH"
    BRAKE = "BRAKE"
    DONE = "DONE"


class AccelerationMission(MissionBehavior):
    LAUNCH_DURATION_S = 3.0

    def __init__(self) -> None:            # must take no arguments
        self._substate = AccelSubState.LAUNCH

    @property
    def name(self) -> str:
        return "Open-loop Acceleration"

    def on_enter(self, context: MissionContext) -> None:
        self._substate = AccelSubState.LAUNCH

    def update(self, context: MissionContext) -> SubStateResult:
        if self._substate == AccelSubState.LAUNCH and context.mission_time > self.LAUNCH_DURATION_S:
            self._substate = AccelSubState.BRAKE
        if self._substate == AccelSubState.BRAKE and abs(context.telemetry.velocity[0]) < 0.05:
            self._substate = AccelSubState.DONE

        throttle = 0.3 if self._substate == AccelSubState.LAUNCH else 0.0
        braking = 0.0 if self._substate == AccelSubState.LAUNCH else 0.8

        return SubStateResult(
            next_substate=self._substate.value,
            mission_finished=self._substate == AccelSubState.DONE,
            actuation_command=ActuationCommandMessage(
                timestamp=context.telemetry.timestamp,
                throttle=throttle,
                braking=braking,
            ),
        )

    def on_exit(self, context: MissionContext, interrupted: bool = False) -> None:
        pass


MissionRegistry.register(Mission.ACCELERATION, AccelerationMission)
```

A few things worth knowing while writing `update()`:

- Use `context.mission_time` and `context.dt` for timing, never `time.time()`. Tests and the mock VCU run faster than real time.
- Sub-state transitions are entirely your business. Level 1 only reads the string you hand back. Keep your sub-states in an `Enum` like Inspection does so typos are caught.
- Always return a command. Returning `actuation_command=None` while driving gets you a service-brake stop.
- `update()` runs inside the tick while holding the node's lock, so the VCU watchdog thread can't run until you return. Keep it fast and never block on I/O.
- Use `emergency_stop=True` when your mission knows something is wrong (lost track, impossible state). Don't raise an exception on purpose; it works, but the log will look like a bug.

**3. Register it by importing it.** Add the import to [level2/missions/\_\_init\_\_.py](level2/missions/__init__.py):

```python
from StateMachine.level2.missions.inspection import InspectionMission
from StateMachine.level2.missions.acceleration import AccelerationMission

__all__ = ["InspectionMission", "AccelerationMission"]
```

Skip this and nothing errors. The registry just won't have your mission, and the car will trip the Level 2 watchdog right after RES Go. (Remember: registering `ACCELERATION` here also means an external planner can no longer drive acceleration.)

**4. Test it** without any pipeline threads, using the mock VCU:

```python
from Pipeline.messages import MacroState, Mission
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.mock_vcu import MockVcuSimulator
from StateMachine.level2.missions.acceleration import AccelerationMission

sm, vcu = Level1StateMachine(mission_behavior=AccelerationMission()), MockVcuSimulator()
vcu.set_readiness_preconditions()
vcu.dmi_mission = Mission.ACCELERATION

for _ in range(260):                 # DS_Ready plus the 5 s lockout
    sm.step(vcu.step())
vcu.res_go_signal, vcu.brake_status = True, False
status, _ = sm.step(vcu.step())
assert status.current_state == MacroState.DS_DRIVING
```

The mock VCU doesn't simulate physics, so velocity stays wherever you set it (`vcu.velocity[0] = 10.0`). [tests/test_inspection_mission.py](tests/test_inspection_mission.py) and [tests/test_mission_completion.py](tests/test_mission_completion.py) are good templates to copy.

## Do I need to send a heartbeat?

Depends who you are.

- **In-process mission:** No. You're called synchronously every tick. If you crash, Level 1 knows immediately.
- **External planner:** Yes, but it isn't a separate message. **Your `ActuationCommandMessage` *is* your heartbeat.** In `DS_Driving`, if the newest command you sent is older than 200 ms (on the telemetry clock), Level 1 trips `WATCHDOG_TIMEOUT_LEVEL2`. `Level2StatusMessage` is not watched; it only carries information.
- **The VCU:** Yes, its telemetry packets are its heartbeat. In `DS_Ready` and `DS_Driving`, a gap over 200 ms trips `WATCHDOG_TIMEOUT_VCU`, caught both by the next late packet and by the watchdog thread if no packet comes at all. In `DS_Off`, `DS_Finished` and manual driving the link isn't watched, because the car is either parked with brakes on or a human is driving.
- **The state machine itself:** It publishes a `heartbeat` counter in every `StateMachineStatusMessage`, incremented once per output. That one is for the VCU to watch (see below).

## How the real VCU fits in

Today, the only VCU in this repo is the fake one in [mock_vcu.py](mock_vcu.py). The real bridge (CAN or serial, on the bench in Phase 3) doesn't exist yet. Whoever writes it needs to honor this contract.

**Into the state machine:** a Pipeline source node that publishes a `VcuTelemetryMessage` at about 50 Hz:

| Field | Meaning |
| --- | --- |
| `timestamp` | Seconds, monotonic, from one consistent clock. This becomes the clock for all of Level 1 and Level 2. |
| `ebs_status` | `UNAVAILABLE`, `ARMED`, or `ACTIVATED` |
| `ebs_sound_active` | EBS warning sound is on (needed to leave `DS_Emergency`) |
| `dsms_on` | Driverless System Master Switch |
| `dbs_check_ok` | Driverless brake system self-check passed |
| `ts_active` | Tractive system is live |
| `dmi_mission` | Mission picked on the DMI |
| `brakes_engaged` | Brakes currently applied |
| `sdc_closed` | Shutdown circuit is closed (healthy) |
| `res_go_signal` | RES Go button is pressed |
| `velocity` | Array, `velocity[0]` is longitudinal speed in m/s. Standstill detection uses it. |

Every field is required on purpose. If the bridge forgets one, construction fails loudly, instead of quietly reporting "SDC closed, brakes engaged" when it doesn't actually know.

**Out of the state machine:** subscribe to the state machine node and you get two messages per tick.

- `ActuationCommandMessage`: the gated steering (rad), throttle (0 to 1) and brake (0 to 1). Always safe to forward as is; the gate already did the work.
- `StateMachineStatusMessage`: the bits the VCU acts on directly:
  - `ebs_trigger_cmd`: True in `DS_Finished` and `DS_Emergency`. Fire / hold EBS.
  - `service_brake_cmd`: True in `DS_Ready` and `DS_Emergency`. Hold the service brake.
  - `dssi_status`: which DSSI light pattern to show.
  - `heartbeat`: increments every tick. The VCU should treat a heartbeat that stops changing as the autonomous computer being dead, and react on its own side (EBS). The state machine can't protect you from itself hanging; only the VCU can.
  - `error_code` / `error_flags`: latched faults (see `DiagnosticErrorCode` in [Pipeline/messages.py](../Pipeline/messages.py)). They stay set until the car goes back to `DS_Off`.

**Getting out of an emergency** requires the VCU to report EBS sound off, DSMS off and brakes released, all at once. And the state machine won't leave `DS_Emergency` until it has published at least one status saying it's in `DS_Emergency`, so the VCU can never miss one.

## The states at a glance

Every transition is listed at the top of [level1_state_machine.py](level1_state_machine.py) and drawn in [flowchart.png](flowchart.png). If you change one, change both. The short version:

| From | To | When |
| --- | --- | --- |
| `DS_Off` | `DS_Ready` | EBS armed, DSMS on, DBS ok, TS on, SDC closed, brakes on, driverless mission selected. Mission is committed here. |
| `DS_Off` | `Manual_Driving` | Manual mission, EBS unavailable, DSMS off, TS on |
| `Manual_Driving` | `DS_Off` | TS off |
| `DS_Ready` | `DS_Driving` | RES Go pressed at least 5 s after entering `DS_Ready`, and everything still healthy. A press during the 5 s must be released and pressed again. |
| `DS_Ready` | `DS_Off` | DSMS off and brakes released |
| `DS_Driving` | `DS_Finished` | Level 2 says finished and the car is at standstill |
| `DS_Finished` | `DS_Off` | DSMS off and brakes released |
| `DS_Emergency` | `DS_Off` | EBS sound off, DSMS off, brakes released |
| `DS_Ready` / `DS_Driving` | `DS_Emergency` | Any fault: SDC open, EBS fired, DSMS or TS dropped, VCU or Level 2 link lost, Level 2 crash or e-stop, no standstill 5 s after finishing |

## Running it

From the repo root:

```bash
python -m StateMachine.mock_vcu
```

runs every scripted scenario (normal mission, premature RES Go, SDC trip, watchdog). Add `--scenario sdc-trip` to run just one.

```bash
python -m StateMachine.mock_vcu -i
```

opens an interactive shell where you flip switches by hand (`ready`, `go`, `sdc trip`, `speed 3`, `step 50`, `help`) and watch the state, faults and gated actuation change. It's the fastest way to build intuition for how the states behave.

```bash
python -m pytest StateMachine/tests
```

runs the test suite.

## Gotchas

- **Two clocks.** Everything in Level 1 and Level 2 uses the telemetry timestamp. Only the node's watchdog thread uses the wall clock. Mixing them is the most likely way to cause a mystery `WATCHDOG_TIMEOUT_LEVEL2`.
- **Unimported mission = unregistered mission.** No error, just a Level 2 watchdog trip after RES Go.
- **The mission is locked at `DS_Ready`.** Changing the DMI after that doesn't switch missions.
- **Missions are rebuilt every run** from the registry (fresh `__init__`). An injected `mission_behavior` is reused across runs, so its `on_enter` has to reset everything.
- **Faults are sticky.** `error_code` accumulates every fault since the last `DS_Off`, not just the first one.
- **Constants live in [constants.py](constants.py).** The steering limit there is an estimate and still needs to be measured on the car as of 11/19/2026. Check the git blame on that one if it really matters 👀
