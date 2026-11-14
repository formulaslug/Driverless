"""Mock Vehicle Control Unit (VCU) simulator and interactive pipeline test harness."""

import time
import argparse
from typing import Optional, Generator, List
import numpy as np

from Pipeline.node import SourceNode, Node
from Pipeline.messages import (
    EbsStatus,
    Mission,
    MacroState,
    DSSIStatus,
    VcuTelemetryMessage,
    StateMachineStatusMessage,
    ActuationCommandMessage,
)
from StateMachine.constants import STATE_MACHINE_DEFAULT_DT_S
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.state_machine_node import StateMachineNode


class MockVcuSimulator:
    """Configurable simulation model of the FSAE Vehicle Control Unit."""

    def __init__(self, initial_time: float = 0.0) -> None:
        self.time: float = initial_time
        self.ebs_status: EbsStatus = EbsStatus.ARMED
        self.ebs_sound_active: bool = False
        self.dsms_status: bool = False
        self.dbs_check_ok: bool = True
        self.ts_status: bool = True
        self.ami_mission: Mission = Mission.INSPECTION
        self.brake_status: bool = True
        self.sdc_status: bool = True
        self.res_go_signal: bool = False
        self.velocity: np.ndarray = np.zeros(3, dtype=np.float64)

    def set_readiness_preconditions(self) -> None:
        """Set all physical preconditions required for AS_Off -> AS_Ready."""
        self.ebs_status = EbsStatus.ARMED
        self.dsms_status = True
        self.dbs_check_ok = True
        self.ts_status = True
        self.ami_mission = Mission.INSPECTION
        self.brake_status = True
        self.sdc_status = True
        self.res_go_signal = False
        self.velocity.fill(0.0)

    def toggle_dsms(self, state: Optional[bool] = None) -> bool:
        """Toggle or set Driverless System Master Switch (DSMS)."""
        self.dsms_status = (not self.dsms_status) if state is None else state
        return self.dsms_status

    def toggle_ebs(self, status: Optional[EbsStatus] = None) -> EbsStatus:
        """Toggle or set Emergency Brake System (EBS) status."""
        if status is not None:
            self.ebs_status = status
        else:
            self.ebs_status = (
                EbsStatus.ACTIVATED if self.ebs_status == EbsStatus.ARMED else EbsStatus.ARMED
            )
        return self.ebs_status

    def trip_sdc(self) -> None:
        """Open the Shutdown Circuit (SDC)."""
        self.sdc_status = False

    def close_sdc(self) -> None:
        """Close/restore the Shutdown Circuit (SDC)."""
        self.sdc_status = True

    def press_res_go(self) -> None:
        """Assert Remote Emergency System (RES) Go / Ready-to-Drive button."""
        self.res_go_signal = True

    def release_res_go(self) -> None:
        """Release Remote Emergency System (RES) Go button."""
        self.res_go_signal = False

    def set_speed(self, vx: float) -> None:
        """Set longitudinal vehicle velocity in m/s."""
        self.velocity[0] = vx

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
        self.simulator: MockVcuSimulator = simulator or MockVcuSimulator()
        self.dt: float = 1.0 / tick_rate_hz
        self.max_ticks: Optional[int] = max_ticks

    def produce(self) -> Generator[VcuTelemetryMessage, None, None]:
        """Generator pumping simulated VCU telemetry packets downstream."""
        ticks = 0
        while not self.stopEvent.is_set():
            if self.max_ticks is not None and ticks >= self.max_ticks:
                break
            msg = self.simulator.step(self.dt)
            yield msg
            ticks += 1
            time.sleep(self.dt)


# ---------------------------------------------------------------------------
# Pre-packaged Automated Scenarios
# ---------------------------------------------------------------------------

def run_normal_mission_scenario(verbose: bool = True) -> bool:
    """Run full normal lifecycle: Off -> Ready (5s timer) -> Driving -> Finished -> Off."""
    sm = Level1StateMachine()
    vcu = MockVcuSimulator(initial_time=0.0)

    if verbose:
        print("[Scenario] 1. Initial State -> Expect AS_Off")
    status, _ = sm.step(vcu.step())
    assert status.current_state == MacroState.AS_OFF

    if verbose:
        print("[Scenario] 2. Activating Preconditions -> Expect AS_Ready")
    vcu.set_readiness_preconditions()
    status, _ = sm.step(vcu.step())
    assert status.current_state == MacroState.AS_READY
    assert status.dssi_status == DSSIStatus.CONTINUOUS_YELLOW

    if verbose:
        print("[Scenario] 3. Enforcing 5.0s lockout timer (stepping 5.1s)...")
    for _ in range(255):  # 255 * 0.02s = 5.1s
        status, _ = sm.step(vcu.step())
        assert status.current_state == MacroState.AS_READY

    if verbose:
        print("[Scenario] 4. Issuing valid RES Go signal -> Expect AS_Driving")
    vcu.res_go_signal = True
    vcu.brake_status = False  # Brakes release for driving
    status, _ = sm.step(vcu.step())
    vcu.res_go_signal = False
    assert status.current_state == MacroState.AS_DRIVING
    assert status.dssi_status == DSSIStatus.FLASHING_YELLOW

    if verbose:
        print("[Scenario] 5. Driving Inspection mission until completion...")
    for _ in range(600):  # Run inspection steps
        status, act = sm.step(vcu.step())
        if status.current_state == MacroState.AS_FINISHED:
            break

    assert status.current_state == MacroState.AS_FINISHED
    assert status.dssi_status == DSSIStatus.FLASHING_BLUE
    assert status.ebs_trigger_cmd is True

    if verbose:
        print("[Scenario] 6. Powering down: DSMS Off & Brakes Released -> Expect AS_Off")
    vcu.dsms_status = False
    vcu.brake_status = False
    status, _ = sm.step(vcu.step())
    assert status.current_state == MacroState.AS_OFF
    if verbose:
        print("[Scenario] SUCCESS: Normal mission sequence completed flawlessly.")
    return True


def run_premature_res_scenario(verbose: bool = True) -> bool:
    """Verify rejection of premature RES Go press during 5.0s lockout."""
    sm = Level1StateMachine()
    vcu = MockVcuSimulator(initial_time=0.0)

    vcu.set_readiness_preconditions()
    status, _ = sm.step(vcu.step())
    assert status.current_state == MacroState.AS_READY

    # Advance 2.0s into ready lockout
    for _ in range(100):
        sm.step(vcu.step())

    if verbose:
        print("[Scenario] Injecting premature RES Go press at t = 2.0s...")
    vcu.res_go_signal = True
    status, _ = sm.step(vcu.step())
    assert status.current_state == MacroState.AS_READY, "Car transitioned prematurely!"
    vcu.res_go_signal = False

    if verbose:
        print("[Scenario] Advancing to t = 5.5s without re-pressing RES Go...")
    for _ in range(175):
        status, _ = sm.step(vcu.step())
        assert status.current_state == MacroState.AS_READY, "Car falsely remembered premature RES Go!"

    if verbose:
        print("[Scenario] Pressing fresh RES Go at t = 5.6s -> Expect AS_Driving")
    vcu.res_go_signal = True
    vcu.brake_status = False
    status, _ = sm.step(vcu.step())
    assert status.current_state == MacroState.AS_DRIVING
    if verbose:
        print("[Scenario] SUCCESS: Premature RES Go correctly rejected.")
    return True


def run_sdc_trip_scenario(verbose: bool = True) -> bool:
    """Verify immediate emergency latching when SDC opens in AS_Driving."""
    sm = Level1StateMachine()
    vcu = MockVcuSimulator(initial_time=0.0)

    vcu.set_readiness_preconditions()
    sm.step(vcu.step())

    # Pass 5s timer
    for _ in range(260):
        sm.step(vcu.step())

    # Launch driving
    vcu.res_go_signal = True
    status, _ = sm.step(vcu.step())
    vcu.res_go_signal = False
    assert status.current_state == MacroState.AS_DRIVING

    if verbose:
        print("[Scenario] Tripping Shutdown Circuit (SDC Open)...")
    vcu.sdc_status = False
    status, act = sm.step(vcu.step())
    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.dssi_status == DSSIStatus.CONTINUOUS_BLUE
    assert act.gated is False
    assert status.ebs_trigger_cmd is True

    if verbose:
        print("[Scenario] Verifying emergency latches while the EBS sound is on...")
    vcu.ebs_sound_active = True
    vcu.dsms_status = False
    vcu.brake_status = False
    status, _ = sm.step(vcu.step())
    assert status.current_state == MacroState.AS_EMERGENCY

    if verbose:
        print("[Scenario] EBS sound off & ASMS off & brakes released -> Expect AS_Off")
    vcu.ebs_sound_active = False
    status, _ = sm.step(vcu.step())
    assert status.current_state == MacroState.AS_OFF
    if verbose:
        print("[Scenario] SUCCESS: SDC emergency trip and latching verified.")
    return True


def run_watchdog_trip_scenario(verbose: bool = True) -> bool:
    """Verify emergency trip when telemetry drops for > 200ms in AS_Ready."""
    sm = Level1StateMachine()
    vcu = MockVcuSimulator(initial_time=0.0)

    vcu.set_readiness_preconditions()
    status, _ = sm.step(vcu.step())
    assert status.current_state == MacroState.AS_READY

    if verbose:
        print("[Scenario] Injecting 300ms telemetry dropout (> 200ms watchdog)...")
    dropped_tel = vcu.step(dt=0.30)
    status, act = sm.step(dropped_tel)

    assert status.current_state == MacroState.AS_EMERGENCY
    assert status.error_code & 1 << 2  # DiagnosticErrorCode.WATCHDOG_TIMEOUT_VCU
    assert act.gated is False
    assert status.ebs_trigger_cmd is True

    if verbose:
        print("[Scenario] SUCCESS: VCU watchdog timeout trip verified.")
    return True


# ---------------------------------------------------------------------------
# Interactive Terminal Mode
# ---------------------------------------------------------------------------

def run_interactive_mode(input_fn=input, output_fn=print) -> None:
    """Run interactive terminal prompt allowing live toggling of VCU signals and stepping the state machine."""
    sm = Level1StateMachine()
    vcu = MockVcuSimulator(initial_time=0.0)

    output_fn("================ Interactive Mock VCU Terminal ================")
    output_fn("Commands:")
    output_fn("  ready              - Apply all AS_Ready preconditions")
    output_fn("  dsms [on|off]      - Toggle or set DSMS switch")
    output_fn("  ebs [armed|act]    - Set EBS to ARMED or ACTIVATED")
    output_fn("  sdc [trip|close]   - Trip (open) or close Shutdown Circuit")
    output_fn("  go [press|rel]     - Press or release RES Go button")
    output_fn("  brakes [on|off]    - Engage or release service brakes")
    output_fn("  speed <vx>         - Set vehicle longitudinal velocity (m/s)")
    output_fn("  mission <name>     - Set mission (inspection, accel, skidpad, etc.)")
    output_fn("  step [n]           - Step state machine n cycles (default 1)")
    output_fn("  sound [on|off]     - EBS warning sound (AS_Emergency exits once it is off)")
    output_fn("  status             - Print current telemetry and state")
    output_fn("  help               - Show this help menu")
    output_fn("  quit / exit        - Exit interactive shell")
    output_fn("===============================================================\n")

    def print_status(status: StateMachineStatusMessage, act: ActuationCommandMessage) -> None:
        err_str = f"0x{status.error_code:04X}" if status.error_code else "NONE"
        output_fn(
            f"[t={status.timestamp:.2f}s] State: {status.current_state.name} | "
            f"DSSI: {status.dssi_status.name} | Error: {err_str} | "
            f"Substate: {status.active_substate} | "
            f"Act: (str={act.steering[0]:.2f}, thr={act.throttle[0]:.2f}, brk={act.braking[0]:.2f}, gated={act.gated})"
        )

    # Initial step
    status, act = sm.step(vcu.step())
    print_status(status, act)

    while True:
        try:
            line = input_fn("mock-vcu> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nExiting interactive shell.")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()
        cmd_args = parts[1:]

        if cmd in ("quit", "exit", "q"):
            output_fn("Exiting interactive shell.")
            break

        elif cmd == "help":
            output_fn("Commands: ready, dsms [on|off], ebs [armed|act], sdc [trip|close], go [press|rel], brakes [on|off], speed <vx>, mission <name>, step [n], sound [on|off], status, quit")

        elif cmd == "ready":
            vcu.set_readiness_preconditions()
            output_fn("Applied AS_Ready preconditions.")
            status, act = sm.step(vcu.step())
            print_status(status, act)

        elif cmd == "dsms":
            if cmd_args:
                val = cmd_args[0].lower() in ("1", "on", "true")
                vcu.toggle_dsms(val)
            else:
                vcu.toggle_dsms()
            output_fn(f"DSMS status: {vcu.dsms_status}")
            status, act = sm.step(vcu.step())
            print_status(status, act)

        elif cmd == "ebs":
            if cmd_args and cmd_args[0].lower() in ("act", "activated", "2"):
                vcu.toggle_ebs(EbsStatus.ACTIVATED)
            else:
                vcu.toggle_ebs(EbsStatus.ARMED)
            output_fn(f"EBS status: {vcu.ebs_status.name}")
            status, act = sm.step(vcu.step())
            print_status(status, act)

        elif cmd == "sdc":
            if cmd_args and cmd_args[0].lower() in ("close", "closed", "1", "true"):
                vcu.close_sdc()
            else:
                vcu.trip_sdc()
            output_fn(f"SDC status: {vcu.sdc_status}")
            status, act = sm.step(vcu.step())
            print_status(status, act)

        elif cmd == "go":
            if cmd_args and cmd_args[0].lower() in ("rel", "release", "0", "false"):
                vcu.release_res_go()
            else:
                vcu.press_res_go()
            output_fn(f"RES Go pressed: {vcu.res_go_signal}")
            status, act = sm.step(vcu.step())
            print_status(status, act)

        elif cmd == "brakes":
            if cmd_args and cmd_args[0].lower() in ("rel", "release", "off", "0", "false"):
                vcu.brake_status = False
            else:
                vcu.brake_status = True
            output_fn(f"Brake status: {vcu.brake_status}")
            status, act = sm.step(vcu.step())
            print_status(status, act)

        elif cmd == "speed":
            if cmd_args:
                try:
                    spd = float(cmd_args[0])
                    vcu.set_speed(spd)
                    output_fn(f"Longitudinal speed set to: {spd:.2f} m/s")
                except ValueError:
                    output_fn(f"Invalid speed: {cmd_args[0]}")
            else:
                output_fn(f"Current speed: {vcu.velocity[0]:.2f} m/s")
            status, act = sm.step(vcu.step())
            print_status(status, act)

        elif cmd == "mission":
            if cmd_args:
                m_str = cmd_args[0].upper()
                try:
                    vcu.ami_mission = Mission[m_str]
                    output_fn(f"Selected mission: {vcu.ami_mission.name}")
                except KeyError:
                    output_fn(f"Unknown mission '{cmd_args[0]}'. Choices: {[m.name for m in Mission]}")
            else:
                output_fn(f"Current mission: {vcu.ami_mission.name}")
            status, act = sm.step(vcu.step())
            print_status(status, act)

        elif cmd == "sound":
            vcu.ebs_sound_active = not (cmd_args and cmd_args[0].lower() in ("off", "0", "false"))
            output_fn(f"EBS sound active: {vcu.ebs_sound_active}")
            status, act = sm.step(vcu.step())
            print_status(status, act)

        elif cmd == "step":
            n = 1
            if cmd_args:
                try:
                    n = max(1, int(cmd_args[0]))
                except ValueError:
                    n = 1
            for _ in range(n):
                status, act = sm.step(vcu.step())
            output_fn(f"Stepped {n} cycles.")
            print_status(status, act)

        elif cmd == "status":
            status, act = sm.step(vcu.step())
            print_status(status, act)

        else:
            output_fn(f"Unrecognized command: '{cmd}'. Type 'help' for available commands.")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mock VCU Test Harness and Scenario Runner")
    parser.add_argument(
        "--scenario",
        choices=["normal", "premature-res", "sdc-trip", "watchdog", "all"],
        default="all",
        help="Automated scenario to execute",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive terminal prompt to toggle VCU signals live",
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive_mode()
        return

    print("=== FSAE Driverless Mock VCU Harness ===")
    if args.scenario in ("normal", "all"):
        run_normal_mission_scenario()
    if args.scenario in ("premature-res", "all"):
        run_premature_res_scenario()
    if args.scenario in ("sdc-trip", "all"):
        run_sdc_trip_scenario()
    if args.scenario in ("watchdog", "all"):
        run_watchdog_trip_scenario()
    print("All requested Mock VCU scenarios passed successfully.")


if __name__ == "__main__":
    main()
