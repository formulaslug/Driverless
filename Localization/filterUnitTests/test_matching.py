import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from filter import ConeFilter

def testColorCompatibility():
    print("\n=== Test 1: Color Compatibility ===")
    coneFilter = ConeFilter()

    blueConf = [0.9, 0.0, 0.0, 0.0]
    yellowConf = [0.0, 0.9, 0.0, 0.0]
    sOrangeConf = [0.0, 0.0, 0.9, 0.0]
    lOrangeConf = [0.0, 0.0, 0.0, 0.9]
    unknownConf = [0.3, 0.2, 0.25, 0.25]

    assert coneFilter.areColorsCompatible(blueConf, blueConf), "Same class should match"
    assert not coneFilter.areColorsCompatible(blueConf, yellowConf), "Different classes should not match"
    assert coneFilter.areColorsCompatible(sOrangeConf, lOrangeConf), "Small and large orange should match"
    assert coneFilter.areColorsCompatible(lOrangeConf, sOrangeConf), "Large and small orange should match"
    assert coneFilter.areColorsCompatible(unknownConf, blueConf), "Unknown should match any class"
    assert coneFilter.areColorsCompatible(yellowConf, unknownConf), "Any class should match unknown"

    print("✓ Color compatibility test passed")

def testPerfectMatch():
    print("\n=== Test 2: Perfect Match ===")
    coneFilter = ConeFilter()

    initialDetections = [
        (10.0, 2.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(initialDetections, 0.0, 0.0, 0.0)

    print(f"After initial update: {len(coneFilter.cones)} cone(s)")
    print(f"  Cone 0: x={coneFilter.cones[0].x:.2f}, y={coneFilter.cones[0].y:.2f}, obs={coneFilter.cones[0].numObservations}")

    sameDetections = [
        (10.0, 2.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(sameDetections, 0.0, 0.0, 0.0)

    print(f"After matching update: {len(coneFilter.cones)} cone(s)")
    print(f"  Cone 0: x={coneFilter.cones[0].x:.2f}, y={coneFilter.cones[0].y:.2f}, obs={coneFilter.cones[0].numObservations}")

    assert len(coneFilter.cones) == 1, "Should still have 1 cone (matched)"
    assert coneFilter.cones[0].numObservations == 2, "Should have 2 observations"
    assert coneFilter.cones[0].age == 0, "Age should be reset to 0"

    print("✓ Perfect match test passed")

def testColorMismatchRejection():
    print("\n=== Test 3: Color Mismatch Rejection ===")
    coneFilter = ConeFilter()

    initialDetections = [
        (10.0, 2.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(initialDetections, 0.0, 0.0, 0.0)

    print(f"Initial cone: blue at x={coneFilter.cones[0].x:.2f}, y={coneFilter.cones[0].y:.2f}")

    differentColorDetections = [
        (10.0, 2.0, 0.0, 0.9, 0.0, 0.0),
    ]
    coneFilter.update(differentColorDetections, 0.0, 0.0, 0.0)

    print(f"After yellow detection at same position: {len(coneFilter.cones)} cone(s)")

    assert len(coneFilter.cones) == 2, "Should have 2 cones (no match due to color)"

    print("✓ Color mismatch rejection test passed")

def testDistanceThreshold():
    print("\n=== Test 4: Distance Threshold ===")
    coneFilter = ConeFilter(matchThreshold=1.0, useMahalanobis=False)

    initialDetections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(initialDetections, 0.0, 0.0, 0.0)

    nearDetection = [
        (10.5, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(nearDetection, 0.0, 0.0, 0.0)
    print(f"After near detection (0.5m away): {len(coneFilter.cones)} cone(s)")
    assert len(coneFilter.cones) == 1, "Should match (within threshold)"

    farDetection = [
        (15.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(farDetection, 0.0, 0.0, 0.0)
    print(f"After far detection (>1m away): {len(coneFilter.cones)} cone(s)")
    assert len(coneFilter.cones) == 2, "Should not match (beyond threshold)"

    print("✓ Distance threshold test passed")

def testOrangeMatching():
    print("\n=== Test 5: Orange Wildcard Matching ===")
    coneFilter = ConeFilter()

    initialDetections = [
        (10.0, 2.0, 0.0, 0.0, 0.9, 0.0),
    ]
    coneFilter.update(initialDetections, 0.0, 0.0, 0.0)

    print(f"Initial cone: small orange")

    largeOrangeDetection = [
        (10.0, 2.0, 0.0, 0.0, 0.0, 0.9),
    ]
    coneFilter.update(largeOrangeDetection, 0.0, 0.0, 0.0)

    print(f"After large orange detection at same position: {len(coneFilter.cones)} cone(s)")
    assert len(coneFilter.cones) == 1, "Small and large orange should match"
    assert coneFilter.cones[0].numObservations == 2, "Should have 2 observations"

    print("✓ Orange wildcard matching test passed")

def testStaleConeRemoval():
    print("\n=== Test 6: Stale Cone Removal ===")
    coneFilter = ConeFilter(maxAge=3)

    initialDetections = [
        (10.0, 2.0, 0.9, 0.0, 0.0, 0.0),
        (15.0, -1.0, 0.0, 0.9, 0.0, 0.0),
    ]
    coneFilter.update(initialDetections, 0.0, 0.0, 0.0)

    print(f"Initial: {len(coneFilter.cones)} cones, ages: {[c.age for c in coneFilter.cones]}")

    for i in range(4):
        coneFilter.update([], 0.0, 0.0, 0.0)
        print(f"Update {i+1}: {len(coneFilter.cones)} cones, ages: {[c.age for c in coneFilter.cones]}")

    assert len(coneFilter.cones) == 0, "All cones should be removed (age > maxAge)"

    print("✓ Stale cone removal test passed")

def testPositionAveraging():
    print("\n=== Test 7: Position Averaging ===")
    coneFilter = ConeFilter(initialVariance=1.0, measurementVariance=0.25)

    initialDetections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(initialDetections, 0.0, 0.0, 0.0)

    print(f"Initial position: x={coneFilter.cones[0].x:.2f}")

    slightlyOffsetDetection = [
        (10.5, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(slightlyOffsetDetection, 0.0, 0.0, 0.0)

    print(f"After matching with x=10.5: x={coneFilter.cones[0].x:.2f}")
    kalmanGain = 1.0 / (1.0 + 0.25)
    expectedX = 10.0 + kalmanGain * 0.5
    assert abs(coneFilter.cones[0].x - expectedX) < 0.01, "Position should follow the Kalman gain"
    assert len(coneFilter.cones) == 1, "Should still be 1 cone (matched)"

    print("✓ Position averaging test passed")

if __name__ == '__main__':
    print("Running Part 2 Data Association Tests")
    print("=" * 50)

    testColorCompatibility()
    testPerfectMatch()
    testColorMismatchRejection()
    testDistanceThreshold()
    testOrangeMatching()
    testStaleConeRemoval()
    testPositionAveraging()

    print("\n" + "=" * 50)
    print("All Part 2 tests passed! ✓")
