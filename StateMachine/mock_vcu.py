"""Mock Vehicle Control Unit (VCU) simulator and interactive pipeline test harness."""

import time
import argparse
from typing import Optional, Generator, List
import numpy as np

import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PIPELINE_DIR = os.path.join(ROOT_DIR, 'Pipeline')
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from node import SourceNode, Node
from messages import (
    EbsStatus,
    Mission,
    MacroState,
    ASSIStatus,
    VcuTelemetryMessage,
    StateMachineStatusMessage,
    ActuationCommandMessage,
)
from StateMachine.constants import STATE_MACHINE_DEFAULT_DT_S
from StateMachine.level1_state_machine import Level1StateMachine
from StateMachine.node import StateMachineNode


class MockVcuSimulator:
    """Configurable simulation model of the FSAE Vehicle Control Unit."""

    def __init__(self, initial_time: float = 0.0) -> None:
        self.time: float = initial_time
        self.ebs_status: EbsStatus = EbsStatus.ARMED
        self.asms_status: bool = False
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
        self.asms_status = True
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
            asms_status=self.asms_status,
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
    assert status.assi_status == ASSIStatus.CONTINUOUS_YELLOW

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
    assert status.assi_status == ASSIStatus.FLASHING_YELLOW

    if verbose:
        print("[Scenario] 5. Driving Inspection mission until completion...")
    for _ in range(600):  # Run inspection steps
        status, act = sm.step(vcu.step())
        if status.current_state == MacroState.AS_FINISHED:
            break

    assert status.current_state == MacroState.AS_FINISHED
    assert status.assi_status == ASSIStatus.FLASHING_BLUE
    assert status.ebs_trigger_cmd is True

    if verbose:
        print("[Scenario] 6. Powering down: ASMS Off & Brakes Released -> Expect AS_Off")
    vcu.asms_status = False
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
    assert status.assi_status == ASSIStatus.CONTINUOUS_BLUE
    assert act.gated is False
    assert status.ebs_trigger_cmd is True

    if verbose:
        print("[Scenario] Verifying emergency latching (cannot reset while ASMS is on)...")
    reset_ok = sm.request_reset(vcu.step())
    assert not reset_ok
    assert sm.current_state == MacroState.AS_EMERGENCY

    if verbose:
        print("[Scenario] Clearing faults and resetting...")
    vcu.sdc_status = True
    vcu.asms_status = False
    vcu.brake_status = False
    vcu.ebs_status = EbsStatus.ARMED
    reset_ok = sm.request_reset(vcu.step())
    assert reset_ok
    assert sm.current_state == MacroState.AS_OFF
    if verbose:
        print("[Scenario] SUCCESS: SDC emergency trip and latching verified.")
    return True


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mock VCU Test Harness and Scenario Runner")
    parser.add_argument(
        "--scenario",
        choices=["normal", "premature-res", "sdc-trip", "all"],
        default="all",
        help="Automated scenario to execute",
    )
    args = parser.parse_args()

    print("=== FSAE Driverless Mock VCU Harness ===")
    if args.scenario in ("normal", "all"):
        run_normal_mission_scenario()
    if args.scenario in ("premature-res", "all"):
        run_premature_res_scenario()
    if args.scenario in ("sdc-trip", "all"):
        run_sdc_trip_scenario()
    print("All requested Mock VCU scenarios passed successfully.")


if __name__ == "__main__":
    main()
