import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from filter import ConeFilter

def testUncertaintyGrowthDuringTransformation():
    print("\n=== Test 1: Uncertainty Growth During Transformation ===")
    coneFilter = ConeFilter(
        xMovementUncertainty=0.1,
        yMovementUncertainty=0.1,
        deltaYawUncertainty=0.01,
        initialVariance=0.25
    )

    detections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(detections, 0.0, 0.0, 0.0)

    initialVarX = coneFilter.cones[0].cov[0, 0]
    initialVarY = coneFilter.cones[0].cov[1, 1]

    print(f"Initial variance: varX={initialVarX:.4f}, varY={initialVarY:.4f}")

    coneFilter.update([], 1.0, 0.0, 0.0)

    afterVarX = coneFilter.cones[0].cov[0, 0]
    afterVarY = coneFilter.cones[0].cov[1, 1]

    print(f"After motion: varX={afterVarX:.4f}, varY={afterVarY:.4f}")

    assert afterVarX > initialVarX, "Variance should increase after motion"
    assert afterVarY > initialVarY, "Variance should increase after motion"

    print("✓ Uncertainty grows during transformation")

def testUncertaintyReductionDuringMeasurement():
    print("\n=== Test 2: Uncertainty Reduction During Measurement ===")
    coneFilter = ConeFilter(
        initialVariance=1.0,
        measurementVariance=0.25
    )

    detections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(detections, 0.0, 0.0, 0.0)

    initialVarX = coneFilter.cones[0].cov[0, 0]
    initialVarY = coneFilter.cones[0].cov[1, 1]

    print(f"Initial variance: varX={initialVarX:.4f}, varY={initialVarY:.4f}")

    sameDetection = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(sameDetection, 0.0, 0.0, 0.0)

    afterVarX = coneFilter.cones[0].cov[0, 0]
    afterVarY = coneFilter.cones[0].cov[1, 1]

    print(f"After matching: varX={afterVarX:.4f}, varY={afterVarY:.4f}")

    assert afterVarX < initialVarX, "Variance should decrease after measurement"
    assert afterVarY < initialVarY, "Variance should decrease after measurement"

    print("✓ Uncertainty reduces during measurement fusion")

def testKalmanGainCalculation():
    print("\n=== Test 3: Kalman Gain Calculation ===")
    coneFilter = ConeFilter(
        xMovementUncertainty=0.0,
        yMovementUncertainty=0.0,
        deltaYawUncertainty=0.0,
        initialVariance=1.0,
        measurementVariance=1.0
    )

    detections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(detections, 0.0, 0.0, 0.0)

    print(f"Initial: x={coneFilter.cones[0].x:.2f}, varX={coneFilter.cones[0].cov[0,0]:.4f}")

    offsetDetection = [
        (12.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(offsetDetection, 0.0, 0.0, 0.0)

    expectedKalmanGain = 1.0 / (1.0 + 1.0)
    expectedX = 10.0 + expectedKalmanGain * (12.0 - 10.0)

    print(f"After fusion: x={coneFilter.cones[0].x:.2f}, expected={expectedX:.2f}")

    assert abs(coneFilter.cones[0].x - expectedX) < 0.01, "Position should match Kalman update"

    expectedVarX = (1 - expectedKalmanGain) * 1.0
    print(f"Variance: varX={coneFilter.cones[0].cov[0,0]:.4f}, expected={expectedVarX:.4f}")

    assert abs(coneFilter.cones[0].cov[0,0] - expectedVarX) < 0.01, "Variance should match Kalman update"

    print("✓ Kalman gain calculation correct")

def testYawUncertaintyContribution():
    print("\n=== Test 4: Yaw Uncertainty Contribution ===")
    coneFilter = ConeFilter(
        xMovementUncertainty=0.05,
        yMovementUncertainty=0.05,
        deltaYawUncertainty=0.1,
        initialVariance=0.25
    )

    nearDetection = [
        (5.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    farDetection = [
        (20.0, 0.0, 0.0, 0.9, 0.0, 0.0),
    ]

    coneFilter.update(nearDetection + farDetection, 0.0, 0.0, 0.0)

    nearTotalBefore = np.trace(coneFilter.cones[0].cov)
    farTotalBefore = np.trace(coneFilter.cones[1].cov)

    print(f"Before rotation: near trace={nearTotalBefore:.4f}, far trace={farTotalBefore:.4f}")

    coneFilter.update([], 0.0, 0.0, 0.1)

    nearTotalAfter = np.trace(coneFilter.cones[0].cov)
    farTotalAfter = np.trace(coneFilter.cones[1].cov)

    print(f"After rotation: near trace={nearTotalAfter:.4f}, far trace={farTotalAfter:.4f}")

    nearGrowth = nearTotalAfter - nearTotalBefore
    farGrowth = farTotalAfter - farTotalBefore

    print(f"Variance growth: near={nearGrowth:.4f}, far={farGrowth:.4f}")

    assert farGrowth > nearGrowth, "Far cones should gain more uncertainty from yaw"

    print("✓ Yaw uncertainty affects distant cones more")

def testMahalanobisVsEuclidean():
    print("\n=== Test 5: Mahalanobis vs Euclidean Distance ===")

    euclideanFilter = ConeFilter(
        matchThreshold=2.0,
        useMahalanobis=False
    )
    mahalanobisFilter = ConeFilter(
        mahalanobisThreshold=3.0,
        useMahalanobis=True
    )

    initialDetections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]

    euclideanFilter.update(initialDetections, 0.0, 0.0, 0.0)
    mahalanobisFilter.update(initialDetections, 0.0, 0.0, 0.0)

    euclideanFilter.cones[0].cov = np.array([[4.0, 0.0], [0.0, 0.25]])
    mahalanobisFilter.cones[0].cov = np.array([[4.0, 0.0], [0.0, 0.25]])

    offsetDetection = [
        (12.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]

    euclideanFilter.update(offsetDetection, 0.0, 0.0, 0.0)
    mahalanobisFilter.update(offsetDetection, 0.0, 0.0, 0.0)

    print(f"Euclidean filter: {len(euclideanFilter.cones)} cones")
    print(f"Mahalanobis filter: {len(mahalanobisFilter.cones)} cones")

    assert len(euclideanFilter.cones) == 2, "2m offset is not strictly inside the 2m Euclidean gate"
    assert len(mahalanobisFilter.cones) == 1, "2m offset is well inside the gate along the uncertain axis"

    print("✓ Mahalanobis distance accounts for uncertainty shape")

def testUncertaintyCycleGrowthAndReduction():
    print("\n=== Test 6: Uncertainty Growth-Reduction Cycle ===")
    coneFilter = ConeFilter(
        xMovementUncertainty=0.1,
        yMovementUncertainty=0.1,
        deltaYawUncertainty=0.01,
        initialVariance=0.25,
        measurementVariance=0.25
    )

    detections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(detections, 0.0, 0.0, 0.0)

    variances = [coneFilter.cones[0].cov[0, 0]]
    print(f"Step 0 (initial): varX={variances[-1]:.4f}")

    for i in range(3):
        coneFilter.update([], 1.0, 0.0, 0.0)
        variances.append(coneFilter.cones[0].cov[0, 0])
        print(f"Step {i+1} (motion): varX={variances[-1]:.4f} (growth)")

        matchingDetection = [
            (coneFilter.cones[0].x, coneFilter.cones[0].y, 0.9, 0.0, 0.0, 0.0),
        ]
        coneFilter.update(matchingDetection, 0.0, 0.0, 0.0)
        variances.append(coneFilter.cones[0].cov[0, 0])
        print(f"Step {i+1} (measurement): varX={variances[-1]:.4f} (reduction)")

    assert variances[1] > variances[0], "Variance should grow after first motion"
    assert variances[2] < variances[1], "Variance should reduce after first measurement"

    print("✓ Uncertainty follows grow-shrink cycle")

def testJacobianRotation():
    print("\n=== Test 7: Jacobian Rotation of Covariance ===")
    coneFilter = ConeFilter(
        xMovementUncertainty=0.0,
        yMovementUncertainty=0.0,
        deltaYawUncertainty=0.0,
        initialVariance=1.0
    )

    detections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(detections, 0.0, 0.0, 0.0)

    print(f"Initial: x={coneFilter.cones[0].x:.2f}, y={coneFilter.cones[0].y:.2f}")
    print(f"Initial variance: varX={coneFilter.cones[0].cov[0,0]:.4f}, varY={coneFilter.cones[0].cov[1,1]:.4f}")

    coneFilter.update([], 0.0, 0.0, np.pi / 2)

    print(f"After 90° rotation: x={coneFilter.cones[0].x:.2f}, y={coneFilter.cones[0].y:.2f}")
    print(f"After variance: varX={coneFilter.cones[0].cov[0,0]:.4f}, varY={coneFilter.cones[0].cov[1,1]:.4f}")

    assert abs(coneFilter.cones[0].cov[0,0] - 1.0) < 0.01, "Variance should be preserved under rotation"
    assert abs(coneFilter.cones[0].cov[1,1] - 1.0) < 0.01, "Variance should be preserved under rotation"

    print("✓ Jacobian correctly rotates covariance matrix")

if __name__ == '__main__':
    print("Running Part 4 Uncertainty Propagation Tests")
    print("=" * 50)

    testUncertaintyGrowthDuringTransformation()
    testUncertaintyReductionDuringMeasurement()
    testKalmanGainCalculation()
    testYawUncertaintyContribution()
    testMahalanobisVsEuclidean()
    testUncertaintyCycleGrowthAndReduction()
    testJacobianRotation()

    print("\n" + "=" * 50)
    print("All Part 4 tests passed! ✓")
