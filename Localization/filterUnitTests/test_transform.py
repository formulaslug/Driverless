import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from filter import ConeFilter

def testPureTranslation():
    print("\n=== Test 1: Pure Translation ===")
    coneFilter = ConeFilter()

    detections = [
        (5.0, 2.0, 0.0, 0.0, 0.9, 0.0),
        (10.0, -3.0, 0.0, 0.85, 0.0, 0.0),
    ]
    coneFilter.addNewCones(detections)

    print(f"Initial cones:")
    for i, cone in enumerate(coneFilter.cones):
        print(f"  Cone {i}: x={cone.x:.2f}, y={cone.y:.2f}")

    dx, dy, dyaw = 2.0, 0.0, 0.0
    coneFilter.transformConeLocations(dx, dy, dyaw)

    print(f"After translation dx={dx}, dy={dy}:")
    for i, cone in enumerate(coneFilter.cones):
        print(f"  Cone {i}: x={cone.x:.2f}, y={cone.y:.2f}")

    assert abs(coneFilter.cones[0].x - 3.0) < 0.01, "Cone 0 x position incorrect"
    assert abs(coneFilter.cones[0].y - 2.0) < 0.01, "Cone 0 y position incorrect"
    assert abs(coneFilter.cones[1].x - 8.0) < 0.01, "Cone 1 x position incorrect"
    assert abs(coneFilter.cones[1].y - (-3.0)) < 0.01, "Cone 1 y position incorrect"

    print("✓ Translation test passed")

def testPureRotation():
    print("\n=== Test 2: Pure Rotation ===")
    coneFilter = ConeFilter()

    detections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.addNewCones(detections)

    print(f"Initial cone: x={coneFilter.cones[0].x:.2f}, y={coneFilter.cones[0].y:.2f}")

    dyaw = np.pi / 2
    coneFilter.transformConeLocations(0.0, 0.0, dyaw)

    print(f"After 90° rotation: x={coneFilter.cones[0].x:.2f}, y={coneFilter.cones[0].y:.2f}")

    assert abs(coneFilter.cones[0].x - 0.0) < 0.01, "Cone x position incorrect after rotation"
    assert abs(coneFilter.cones[0].y - (-10.0)) < 0.01, "Cone y position incorrect after rotation"

    print("✓ Rotation test passed")

def testCombinedMotion():
    print("\n=== Test 3: Combined Translation + Rotation ===")
    coneFilter = ConeFilter()

    detections = [
        (5.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.addNewCones(detections)

    print(f"Initial cone: x={coneFilter.cones[0].x:.2f}, y={coneFilter.cones[0].y:.2f}")

    dx, dy, dyaw = 1.0, 0.5, np.pi / 4
    coneFilter.transformConeLocations(dx, dy, dyaw)

    print(f"After combined motion: x={coneFilter.cones[0].x:.2f}, y={coneFilter.cones[0].y:.2f}")

    cosYaw = np.cos(-dyaw)
    sinYaw = np.sin(-dyaw)
    dxRot = dx * cosYaw - dy * sinYaw
    dyRot = dx * sinYaw + dy * cosYaw
    expectedX = 5.0 * cosYaw - 0.0 * sinYaw - dxRot
    expectedY = 5.0 * sinYaw + 0.0 * cosYaw - dyRot

    print(f"Expected: x={expectedX:.2f}, y={expectedY:.2f}")

    assert abs(coneFilter.cones[0].x - expectedX) < 0.01, "Cone x position incorrect"
    assert abs(coneFilter.cones[0].y - expectedY) < 0.01, "Cone y position incorrect"

    print("✓ Combined motion test passed")

def testUpdateMethod():
    print("\n=== Test 4: Full Update Method ===")
    coneFilter = ConeFilter()

    initialDetections = [
        (10.0, 2.0, 0.9, 0.0, 0.0, 0.0),
        (15.0, -1.0, 0.0, 0.85, 0.0, 0.0),
    ]

    coneFilter.update(initialDetections, 0.0, 0.0, 0.0)
    print(f"After initial update: {len(coneFilter.cones)} cones")

    newDetections = [
        (20.0, 3.0, 0.0, 0.0, 0.8, 0.0),
    ]

    coneFilter.update(newDetections, 2.0, 0.0, 0.0)
    print(f"After second update: {len(coneFilter.cones)} cones")

    assert len(coneFilter.cones) == 3, "Should have 3 cones total"
    assert abs(coneFilter.cones[0].x - 8.0) < 0.01, "First cone should have moved"
    assert abs(coneFilter.cones[2].x - 20.0) < 0.01, "New cone position incorrect"

    print("✓ Update method test passed")

if __name__ == '__main__':
    print("Running Part 1 Transformation Tests")
    print("=" * 50)

    testPureTranslation()
    testPureRotation()
    testCombinedMotion()
    testUpdateMethod()

    print("\n" + "=" * 50)
    print("All Part 1 tests passed! ✓")
