import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from filter import ConeFilter

def testStaticVisualization():
    print("\n=== Test 1: Static Visualization ===")
    coneFilter = ConeFilter()

    detections = [
        (5.0, 2.0, 0.9, 0.0, 0.0, 0.0),
        (8.0, -1.5, 0.0, 0.9, 0.0, 0.0),
        (12.0, 3.0, 0.0, 0.0, 0.9, 0.0),
        (15.0, 0.0, 0.0, 0.0, 0.0, 0.9),
        (10.0, -3.0, 0.9, 0.0, 0.0, 0.0),
        (18.0, 2.5, 0.0, 0.9, 0.0, 0.0),
        (20.0, -2.0, 0.0, 0.0, 0.9, 0.0),
        (7.0, 1.0, 0.0, 0.0, 0.0, 0.9),
        (14.0, -1.0, 0.9, 0.0, 0.0, 0.0),
        (16.0, 1.5, 0.0, 0.9, 0.0, 0.0),
    ]

    coneFilter.update(detections, 0.0, 0.0, 0.0)

    print(f"Created {len(coneFilter.cones)} cones")

    outputPath = os.path.join(os.path.dirname(__file__), '..', 'output', 'test_static_visualization.png')
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)

    coneFilter.visualize(savePath=outputPath, showPlot=False)

    print(f"✓ Saved static visualization to {outputPath}")

def testAnimationSequence():
    print("\n=== Test 2: Animation Sequence ===")
    coneFilter = ConeFilter()

    initialDetections = [
        (10.0, 2.0, 0.9, 0.0, 0.0, 0.0),
        (15.0, -1.0, 0.0, 0.9, 0.0, 0.0),
        (20.0, 0.0, 0.0, 0.0, 0.9, 0.0),
    ]

    coneFilter.update(initialDetections, 0.0, 0.0, 0.0)

    outputDir = os.path.join(os.path.dirname(__file__), '..', 'output', 'animation')
    os.makedirs(outputDir, exist_ok=True)

    coneFilter.visualize(savePath=os.path.join(outputDir, 'frame_000.png'), showPlot=False)
    print(f"Frame 0: {len(coneFilter.cones)} cones")

    coneFilter.update([], 2.0, 0.0, 0.0)
    coneFilter.visualize(savePath=os.path.join(outputDir, 'frame_001.png'), showPlot=False)
    print(f"Frame 1: {len(coneFilter.cones)} cones (vehicle moved 2m forward)")

    newDetections = [
        (10.0, 2.0, 0.9, 0.0, 0.0, 0.0),
        (15.0, -1.0, 0.0, 0.9, 0.0, 0.0),
        (20.0, 0.0, 0.0, 0.0, 0.9, 0.0),
        (25.0, 1.5, 0.0, 0.0, 0.0, 0.9),
    ]
    coneFilter.update(newDetections, 2.0, 0.0, 0.0)
    coneFilter.visualize(savePath=os.path.join(outputDir, 'frame_002.png'), showPlot=False)
    print(f"Frame 2: {len(coneFilter.cones)} cones (added new detection)")

    coneFilter.update([], 2.0, 0.0, np.pi / 8)
    coneFilter.visualize(savePath=os.path.join(outputDir, 'frame_003.png'), showPlot=False)
    print(f"Frame 3: {len(coneFilter.cones)} cones (vehicle turned)")

    print(f"✓ Saved animation sequence to {outputDir}/")

def testHighUncertainty():
    print("\n=== Test 3: High Uncertainty Visualization ===")
    coneFilter = ConeFilter(initialVariance=2.0)

    detections = [
        (10.0, 0.0, 0.9, 0.0, 0.0, 0.0),
        (15.0, 2.0, 0.0, 0.9, 0.0, 0.0),
        (20.0, -2.0, 0.0, 0.0, 0.9, 0.0),
    ]

    coneFilter.update(detections, 0.0, 0.0, 0.0)

    print(f"Created {len(coneFilter.cones)} cones with high initial variance")

    outputPath = os.path.join(os.path.dirname(__file__), '..', 'output', 'test_high_uncertainty.png')
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)

    coneFilter.visualize(savePath=outputPath, showPlot=False)

    print(f"✓ Saved high uncertainty visualization to {outputPath}")
    print("  Check that ellipses are visibly wider than normal")

def testUnknownCones():
    print("\n=== Test 4: Unknown Cone Visualization ===")
    coneFilter = ConeFilter()

    detections = [
        (10.0, 2.0, 0.3, 0.25, 0.25, 0.2),
        (15.0, -1.0, 0.0, 0.9, 0.0, 0.0),
        (20.0, 0.0, 0.2, 0.3, 0.3, 0.2),
    ]

    coneFilter.update(detections, 0.0, 0.0, 0.0)

    print(f"Created {len(coneFilter.cones)} cones (2 unknown, 1 yellow)")

    outputPath = os.path.join(os.path.dirname(__file__), '..', 'output', 'test_unknown_cones.png')
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)

    coneFilter.visualize(savePath=outputPath, showPlot=False)

    print(f"✓ Saved unknown cones visualization to {outputPath}")
    print("  Check that cones with low confidence appear gray")

def testEmptyMap():
    print("\n=== Test 5: Empty Map Visualization ===")
    coneFilter = ConeFilter()

    outputPath = os.path.join(os.path.dirname(__file__), '..', 'output', 'test_empty_map.png')
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)

    coneFilter.visualize(savePath=outputPath, showPlot=False)

    print(f"✓ Saved empty map visualization to {outputPath}")
    print("  Check that only vehicle triangle is visible")

if __name__ == '__main__':
    print("Running Part 3 Visualization Tests")
    print("=" * 50)
    print("NOTE: These tests generate images for manual inspection")
    print("=" * 50)

    testStaticVisualization()
    testAnimationSequence()
    testHighUncertainty()
    testUnknownCones()
    testEmptyMap()

    print("\n" + "=" * 50)
    print("All Part 3 visualizations generated! ✓")
    print("\nCheck the output directory:")
    print("  Localization/output/")
