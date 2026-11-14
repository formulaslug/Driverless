"""Mock Vehicle Control Unit (VCU) simulator and interactive test harness.

Run from the repository root:
    python -m StateMachine.mock_vcu                 # all automated scenarios
    python -m StateMachine.mock_vcu --scenario sdc-trip
    python -m StateMachine.mock_vcu -i              # interactive shell
"""

import argparse
import time
from typing import Callable, Generator, Optional

import numpy as np

from Pipeline.node import SourceNode
from Pipeline.messages import (
    ActuationCommandMessage,
    DiagnosticErrorCode,
    DSSIStatus,
    EbsStatus,
    MacroState,
    Mission,
    StateMachineStatusMessage,
    VcuTelemetryMessage,
)
from StateMachine.constants import STATE_MACHINE_DEFAULT_DT_S
from StateMachine.level1_state_machine import Level1StateMachine


class ScenarioFailed(RuntimeError):
    """Raised when an automated scenario sees the wrong state. Unlike assert, survives python -O."""


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioFailed(message)


class MockVcuSimulator:
    """Configurable simulation model of the FSAE Vehicle Control Unit."""

    def __init__(self, initial_time: float = 0.0) -> None:
        self.time = initial_time
        self.ebs_status = EbsStatus.ARMED
        self.ebs_sound_active = False
        self.dsms_status = False
        self.dbs_check_ok = True
        self.ts_status = True
        self.ami_mission = Mission.INSPECTION
        self.brake_status = True
        self.sdc_status = True
        self.res_go_signal = False
        self.velocity = np.zeros(3, dtype=np.float64)

    def set_readiness_preconditions(self) -> None:
        """Set all physical preconditions required for AS_Off -> AS_Ready."""
        self.ebs_status = EbsStatus.ARMED
        self.ebs_sound_active = False
        self.dsms_status = True
        self.dbs_check_ok = True
        self.ts_status = True
        self.ami_mission = Mission.INSPECTION
        self.brake_status = True
        self.sdc_status = True
        self.res_go_signal = False
        self.velocity.fill(0.0)

    def step(self, dt: float = STATE_MACHINE_DEFAULT_DT_S) -> VcuTelemetryMessage:
        """Advance simulator time and return a fresh telemetry snapshot."""
        self.time += dt
        return VcuTelemetryMessage(
            timestamp=self.time,
            ebs_status=self.ebs_status,
            ebs_sound_active=self.ebs_sound_active,
            dsms_status=self.dsms_status,
            dbs_check_ok=self.dbs_check_ok,
            ts_status=self.ts_status,
            ami_mission=self.ami_mission,
            brake_status=self.brake_status,
            sdc_status=self.sdc_status,
            res_go_signal=self.res_go_signal,
            velocity=self.velocity.copy(),
        )


class MockVcuNode(SourceNode):
    """Pipeline SourceNode that periodically publishes simulated VCU telemetry."""

    def __init__(
        self,
        name: str = "mock_vcu",
        simulator: Optional[MockVcuSimulator] = None,
        tick_rate_hz: float = 50.0,
        max_ticks: Optional[int] = None,
    ) -> None:
        super().__init__(name)
        self.simulator = simulator or MockVcuSimulator()
        self.dt = 1.0 / tick_rate_hz
        self.max_ticks = max_ticks

    def produce(self) -> Generator[VcuTelemetryMessage, None, None]:
        ticks = 0
        while not self.stopEvent.is_set():
            if self.max_ticks is not None and ticks >= self.max_ticks:
                break
            yield self.simulator.step(self.dt)
            ticks += 1
            time.sleep(self.dt)


# ---------------------------------------------------------------------------
# Automated scenarios
# ---------------------------------------------------------------------------

def _run_ticks(sm: Level1StateMachine, vcu: MockVcuSimulator, n: int) -> StateMachineStatusMessage:
    status = None
    for _ in range(n):
        status, _ = sm.step(vcu.step())
    return status


def _ready_then_go(sm: Level1StateMachine, vcu: MockVcuSimulator, log: Callable[[str], None]) -> None:
    """AS_Off -> AS_Ready -> (5.1 s) -> AS_Driving."""
    vcu.set_readiness_preconditions()
    status = _run_ticks(sm, vcu, 1)
    _expect(status.current_state == MacroState.AS_READY, f"expected AS_READY, got {status.current_state.name}")
    _expect(status.dssi_status == DSSIStatus.CONTINUOUS_YELLOW, "DSSI should be continuous yellow in AS_Ready")

    log("Waiting out the 5.0 s AS_Ready lockout")
    status = _run_ticks(sm, vcu, 255)
    _expect(status.current_state == MacroState.AS_READY, "left AS_Ready during the lockout")

    log("RES Go -> expect AS_Driving")
    vcu.res_go_signal = True
    vcu.brake_status = False
    status = _run_ticks(sm, vcu, 1)
    vcu.res_go_signal = False
    _expect(status.current_state == MacroState.AS_DRIVING, f"expected AS_DRIVING, got {status.current_state.name}")


def run_normal_mission_scenario(log: Callable[[str], None] = print) -> None:
    """Off -> Ready (5 s timer) -> Driving (inspection) -> Finished -> Off."""
    sm, vcu = Level1StateMachine(), MockVcuSimulator()
    _ready_then_go(sm, vcu, log)

    log("Running the inspection mission until AS_Finished")
    for _ in range(600):
        status, _ = sm.step(vcu.step())
        if status.current_state == MacroState.AS_FINISHED:
            break
    _expect(status.current_state == MacroState.AS_FINISHED, f"expected AS_FINISHED, got {status.current_state.name}")
    _expect(status.ebs_trigger_cmd, "EBS must be commanded in AS_Finished")

    log("ASMS off & brakes released -> expect AS_Off")
    vcu.dsms_status = False
    vcu.brake_status = False
    status = _run_ticks(sm, vcu, 1)
    _expect(status.current_state == MacroState.AS_OFF, f"expected AS_OFF, got {status.current_state.name}")
    log("SUCCESS: normal mission")


def run_premature_res_scenario(log: Callable[[str], None] = print) -> None:
    """A RES Go press during the 5 s lockout must not launch the car later."""
    sm, vcu = Level1StateMachine(), MockVcuSimulator()
    vcu.set_readiness_preconditions()
    _run_ticks(sm, vcu, 101)

    log("Premature RES Go at t = 2.0 s")
    vcu.res_go_signal = True
    status = _run_ticks(sm, vcu, 1)
    vcu.res_go_signal = False
    _expect(status.current_state == MacroState.AS_READY, "car launched during the lockout")

    log("Advancing to t = 5.5 s without pressing again")
    status = _run_ticks(sm, vcu, 175)
    _expect(status.current_state == MacroState.AS_READY, "car launched from a remembered premature press")

    log("Fresh RES Go -> expect AS_Driving")
    vcu.res_go_signal = True
    vcu.brake_status = False
    status = _run_ticks(sm, vcu, 1)
    _expect(status.current_state == MacroState.AS_DRIVING, f"expected AS_DRIVING, got {status.current_state.name}")
    log("SUCCESS: premature RES Go rejected")


def run_sdc_trip_scenario(log: Callable[[str], None] = print) -> None:
    """SDC opens in AS_Driving -> AS_Emergency, then the flowchart exit back to AS_Off."""
    sm, vcu = Level1StateMachine(), MockVcuSimulator()
    _ready_then_go(sm, vcu, log)

    log("Opening the shutdown circuit")
    vcu.sdc_status = False
    vcu.ebs_status = EbsStatus.ACTIVATED
    vcu.ebs_sound_active = True
    status, act = sm.step(vcu.step())
    _expect(status.current_state == MacroState.AS_EMERGENCY, f"expected AS_EMERGENCY, got {status.current_state.name}")
    _expect(status.dssi_status == DSSIStatus.CONTINUOUS_BLUE, "DSSI should be continuous blue in AS_Emergency")
    _expect(status.ebs_trigger_cmd and not act.gated, "EBS must be commanded and actuation gated off")

    log("ASMS off & brakes released, EBS sound still on -> expect AS_Emergency")
    vcu.dsms_status = False
    vcu.brake_status = False
    status = _run_ticks(sm, vcu, 1)
    _expect(status.current_state == MacroState.AS_EMERGENCY, "left AS_Emergency while the EBS sound was on")

    log("EBS sound off -> expect AS_Off")
    vcu.ebs_sound_active = False
    status = _run_ticks(sm, vcu, 1)
    _expect(status.current_state == MacroState.AS_OFF, f"expected AS_OFF, got {status.current_state.name}")
    log("SUCCESS: SDC trip and emergency exit")


def run_watchdog_trip_scenario(log: Callable[[str], None] = print) -> None:
    """A 300 ms telemetry gap in AS_Ready trips the VCU watchdog."""
    sm, vcu = Level1StateMachine(), MockVcuSimulator()
    vcu.set_readiness_preconditions()
    _run_ticks(sm, vcu, 1)

    log("Injecting a 300 ms telemetry dropout")
    status, act = sm.step(vcu.step(dt=0.30))
    _expect(status.current_state == MacroState.AS_EMERGENCY, f"expected AS_EMERGENCY, got {status.current_state.name}")
    _expect(bool(status.error_code & DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU), "missing WATCHDOG_TIMEOUT_VCU")
    _expect(status.ebs_trigger_cmd and not act.gated, "EBS must be commanded and actuation gated off")
    log("SUCCESS: VCU watchdog")


SCENARIOS = {
    "normal": run_normal_mission_scenario,
    "premature-res": run_premature_res_scenario,
    "sdc-trip": run_sdc_trip_scenario,
    "watchdog": run_watchdog_trip_scenario,
}


# ---------------------------------------------------------------------------
# Interactive terminal mode
# ---------------------------------------------------------------------------

HELP_TEXT = """Commands (each one applies the change, then steps the state machine once):
  ready                     - Apply all AS_Ready preconditions
  dsms [on|off]             - Driverless System Master Switch (ASMS)
  ebs [armed|act|unavail]   - EBS status (unavail + mission manual + ts on -> Manual Driving)
  sound [on|off]            - EBS warning sound (AS_Emergency exits once it is off)
  sdc [trip|close]          - Open or close the shutdown circuit
  ts [on|off]               - Tractive system
  dbs [ok|fail]             - DBS check result
  go [press|rel]            - Press or release RES Go
  brakes [on|off]           - Brake status
  speed <vx>                - Longitudinal velocity in m/s
  mission <name>            - AMI mission (manual, inspection, acceleration, skidpad, ...)
  step [n]                  - Step n cycles without changing anything (default 1)
  status                    - Step once and print
  help                      - Show this help
  quit / exit               - Leave the shell"""

_OFF_WORDS = ("off", "0", "false", "rel", "release", "trip", "fail")


def _is_on(args) -> bool:
    """'cmd' or 'cmd on' -> True; 'cmd off' (or rel/trip/fail/0/false) -> False."""
    return not (args and args[0].lower() in _OFF_WORDS)


def run_interactive_mode(input_fn=input, output_fn=print) -> None:
    """Interactive prompt for toggling VCU signals and stepping the state machine."""
    sm, vcu = Level1StateMachine(), MockVcuSimulator()

    def show(status: StateMachineStatusMessage, act: ActuationCommandMessage) -> None:
        err = repr(DiagnosticErrorCode(status.error_code)) if status.error_code else "NONE"
        output_fn(
            f"[t={status.timestamp:.2f}s] State: {status.current_state.name} | "
            f"DSSI: {status.dssi_status.name} | Error: {err} | "
            f"Mission: {status.active_mission.name} | Substate: {status.active_substate} | "
            f"Act: (str={act.steering[0]:.2f}, thr={act.throttle[0]:.2f}, "
            f"brk={act.braking[0]:.2f}, gated={act.gated})"
        )

    output_fn(HELP_TEXT)
    show(*sm.step(vcu.step()))

    while True:
        try:
            line = input_fn("mock-vcu> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nExiting interactive shell.")
            return
        if not line:
            continue
        cmd, *args = line.split()
        cmd = cmd.lower()

        if cmd in ("quit", "exit", "q"):
            output_fn("Exiting interactive shell.")
            return
        if cmd == "help":
            output_fn(HELP_TEXT)
            continue
        if cmd == "step":
            n = max(1, int(args[0])) if args and args[0].isdigit() else 1
            for _ in range(n - 1):
                sm.step(vcu.step())
            show(*sm.step(vcu.step()))
            output_fn(f"Stepped {n} cycles.")
            continue

        if cmd == "ready":
            vcu.set_readiness_preconditions()
            output_fn("Applied AS_Ready preconditions.")
        elif cmd == "dsms":
            vcu.dsms_status = _is_on(args)
            output_fn(f"DSMS: {vcu.dsms_status}")
        elif cmd == "ebs":
            choice = args[0].lower() if args else "armed"
            vcu.ebs_status = {"act": EbsStatus.ACTIVATED, "unavail": EbsStatus.UNAVAILABLE}.get(choice, EbsStatus.ARMED)
            output_fn(f"EBS: {vcu.ebs_status.name}")
        elif cmd == "sound":
            vcu.ebs_sound_active = _is_on(args)
            output_fn(f"EBS sound active: {vcu.ebs_sound_active}")
        elif cmd == "sdc":
            vcu.sdc_status = _is_on(args)  # "sdc trip" opens it, "sdc close" closes it
            output_fn(f"SDC closed: {vcu.sdc_status}")
        elif cmd == "ts":
            vcu.ts_status = _is_on(args)
            output_fn(f"TS active: {vcu.ts_status}")
        elif cmd == "dbs":
            vcu.dbs_check_ok = _is_on(args)
            output_fn(f"DBS check ok: {vcu.dbs_check_ok}")
        elif cmd == "go":
            vcu.res_go_signal = _is_on(args)
            output_fn(f"RES Go pressed: {vcu.res_go_signal}")
        elif cmd == "brakes":
            vcu.brake_status = _is_on(args)
            output_fn(f"Brakes engaged: {vcu.brake_status}")
        elif cmd == "speed":
            try:
                vcu.velocity[0] = float(args[0])
                output_fn(f"Longitudinal speed set to: {vcu.velocity[0]:.2f} m/s")
            except (IndexError, ValueError):
                output_fn(f"Current speed: {vcu.velocity[0]:.2f} m/s (usage: speed <vx>)")
        elif cmd == "mission":
            try:
                vcu.ami_mission = Mission[args[0].upper()]
                output_fn(f"Selected mission: {vcu.ami_mission.name}")
            except (IndexError, KeyError):
                output_fn(f"Choices: {[m.name.lower() for m in Mission]}")
        elif cmd != "status":
            output_fn(f"Unrecognized command: '{cmd}'. Type 'help' for available commands.")
            continue

        show(*sm.step(vcu.step()))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mock VCU test harness and scenario runner")
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive shell to toggle VCU signals")
    args = parser.parse_args()

    if args.interactive:
        run_interactive_mode()
        return

    print("=== FSAE Driverless Mock VCU Harness ===")
    for name, scenario in SCENARIOS.items():
        if args.scenario in (name, "all"):
            print(f"--- {name}")
            scenario(lambda msg: print(f"[{name}] {msg}"))
    print("All requested Mock VCU scenarios passed.")


if __name__ == "__main__":
    main()
