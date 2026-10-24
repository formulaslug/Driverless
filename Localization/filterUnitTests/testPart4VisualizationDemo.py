import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from filter import ConeFilter

def demonstrateUncertaintyEvolution():
    print("\n=== Uncertainty Evolution Demonstration ===")
    coneFilter = ConeFilter(
        xMovementUncertainty=0.1,
        yMovementUncertainty=0.1,
        deltaYawUncertainty=0.02,
        initialVariance=0.5,
        measurementVariance=0.3
    )

    initialDetections = [
        (10.0, 2.0, 0.9, 0.0, 0.0, 0.0),
        (15.0, -1.0, 0.0, 0.9, 0.0, 0.0),
        (20.0, 0.0, 0.0, 0.0, 0.9, 0.0),
    ]

    outputDir = os.path.join(os.path.dirname(__file__), '..', 'output', 'uncertainty_demo')
    os.makedirs(outputDir, exist_ok=True)

    coneFilter.update(initialDetections, 0.0, 0.0, 0.0)
    coneFilter.visualize(savePath=os.path.join(outputDir, 'step_0_initial.png'), showPlot=False)
    print(f"Step 0: Initial detections (small ellipses)")

    for i in range(3):
        coneFilter.update([], 2.0, 0.0, 0.0)
        coneFilter.visualize(savePath=os.path.join(outputDir, f'step_{i+1}_motion.png'), showPlot=False)
        print(f"Step {i+1}: Motion only (ellipses grow)")

    matchingDetections = [
        (coneFilter.cones[0].x, coneFilter.cones[0].y, 0.9, 0.0, 0.0, 0.0),
        (coneFilter.cones[1].x, coneFilter.cones[1].y, 0.0, 0.9, 0.0, 0.0),
        (coneFilter.cones[2].x, coneFilter.cones[2].y, 0.0, 0.0, 0.9, 0.0),
    ]
    coneFilter.update(matchingDetections, 0.0, 0.0, 0.0)
    coneFilter.visualize(savePath=os.path.join(outputDir, 'step_4_measurement.png'), showPlot=False)
    print(f"Step 4: Measurements (ellipses shrink)")

    print(f"\n✓ Saved uncertainty evolution to {outputDir}/")
    print("  Observe: ellipses grow during motion, shrink during measurement")

def demonstrateYawUncertaintyEffect():
    print("\n=== Yaw Uncertainty Effect Demonstration ===")
    coneFilter = ConeFilter(
        xMovementUncertainty=0.05,
        yMovementUncertainty=0.05,
        deltaYawUncertainty=0.1,
        initialVariance=0.25
    )

    detections = [
        (5.0, 0.0, 0.9, 0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0, 0.9, 0.0, 0.0),
        (20.0, 0.0, 0.0, 0.0, 0.9, 0.0),
    ]

    outputDir = os.path.join(os.path.dirname(__file__), '..', 'output', 'yaw_uncertainty_demo')
    os.makedirs(outputDir, exist_ok=True)

    coneFilter.update(detections, 0.0, 0.0, 0.0)
    coneFilter.visualize(savePath=os.path.join(outputDir, 'before_rotation.png'), showPlot=False)
    print(f"Before rotation: all cones have same small uncertainty")

    coneFilter.update([], 0.0, 0.0, 0.2)
    coneFilter.visualize(savePath=os.path.join(outputDir, 'after_rotation.png'), showPlot=False)
    print(f"After rotation: far cone has much larger uncertainty")

    print(f"\n✓ Saved yaw uncertainty demo to {outputDir}/")
    print("  Observe: far cones gain more uncertainty during rotation")

def demonstrateMahalanobisMatching():
    print("\n=== Mahalanobis Matching Demonstration ===")

    euclideanFilter = ConeFilter(
        matchThreshold=2.0,
        useMahalanobis=False,
        initialVariance=0.5
    )
    mahalanobisFilter = ConeFilter(
        mahalanobisThreshold=3.0,
        useMahalanobis=True,
        initialVariance=0.5
    )

    outputDir = os.path.join(os.path.dirname(__file__), '..', 'output', 'mahalanobis_demo')
    os.makedirs(outputDir, exist_ok=True)

    initialDetections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]

    euclideanFilter.update(initialDetections, 0.0, 0.0, 0.0)
    mahalanobisFilter.update(initialDetections, 0.0, 0.0, 0.0)

    euclideanFilter.cones[0].cov = np.array([[4.0, 0.0], [0.0, 0.25]])
    mahalanobisFilter.cones[0].cov = np.array([[4.0, 0.0], [0.0, 0.25]])

    euclideanFilter.visualize(savePath=os.path.join(outputDir, 'initial_elongated.png'), showPlot=False)
    print(f"Initial: cone with elongated ellipse (varX=4.0, varY=0.25)")

    offsetDetection = [
        (12.0, 0.0, 0.9, 0.0, 0.0, 0.0),
    ]

    euclideanFilter.update(offsetDetection, 0.0, 0.0, 0.0)
    euclideanFilter.visualize(savePath=os.path.join(outputDir, 'euclidean_result.png'), showPlot=False)
    print(f"Euclidean: {len(euclideanFilter.cones)} cones (no match, added new)")

    mahalanobisFilter.update(offsetDetection, 0.0, 0.0, 0.0)
    mahalanobisFilter.visualize(savePath=os.path.join(outputDir, 'mahalanobis_result.png'), showPlot=False)
    print(f"Mahalanobis: {len(mahalanobisFilter.cones)} cones (matched, accounted for elongation)")

    print(f"\n✓ Saved Mahalanobis demo to {outputDir}/")
    print("  Observe: Mahalanobis correctly matches despite elongated uncertainty")

if __name__ == '__main__':
    print("Generating Part 4 Uncertainty Visualizations")
    print("=" * 50)

    demonstrateUncertaintyEvolution()
    demonstrateYawUncertaintyEffect()
    demonstrateMahalanobisMatching()

    print("\n" + "=" * 50)
    print("All Part 4 visualizations generated! ✓")
