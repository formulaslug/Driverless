import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from filter import ConeFilter

def demonstrateColorConfidenceEvolution():
    print("\n=== Color Confidence Evolution Demonstration ===")
    coneFilter = ConeFilter()

    ambiguousDetection = [
        (10.0, 2.0, 0.4, 0.6, 0.0, 0.0),
        (15.0, -1.0, 0.0, 0.0, 0.5, 0.5),
    ]

    outputDir = os.path.join(os.path.dirname(__file__), '..', 'output', 'color_evolution_demo')
    os.makedirs(outputDir, exist_ok=True)

    coneFilter.update(ambiguousDetection, 0.0, 0.0, 0.0)
    coneFilter.visualize(savePath=os.path.join(outputDir, 'step_0_ambiguous.png'), showPlot=False)

    print(f"Step 0 (ambiguous):")
    print(f"  Cone 0: blue={coneFilter.cones[0].colorConfidences[0]:.2f}, yellow={coneFilter.cones[0].colorConfidences[1]:.2f}")
    print(f"  Cone 1: sOrange={coneFilter.cones[1].colorConfidences[2]:.2f}, lOrange={coneFilter.cones[1].colorConfidences[3]:.2f}")

    for i in range(5):
        refinedDetections = [
            (10.0, 2.0, 0.9, 0.1, 0.0, 0.0),
            (15.0, -1.0, 0.0, 0.0, 0.1, 0.9),
        ]
        coneFilter.update(refinedDetections, 0.0, 0.0, 0.0)
        coneFilter.visualize(savePath=os.path.join(outputDir, f'step_{i+1}_refined.png'), showPlot=False)

        print(f"Step {i+1} (refined):")
        print(f"  Cone 0: blue={coneFilter.cones[0].colorConfidences[0]:.2f}, yellow={coneFilter.cones[0].colorConfidences[1]:.2f}")
        print(f"  Cone 1: sOrange={coneFilter.cones[1].colorConfidences[2]:.2f}, lOrange={coneFilter.cones[1].colorConfidences[3]:.2f}")

    print(f"\n✓ Saved color evolution to {outputDir}/")
    print("  Observe: colors become more confident over time")

def demonstrateColorSimilarityMatching():
    print("\n=== Color Similarity Matching Demonstration ===")

    strictFilter = ConeFilter(colorMatchThreshold=0.7)
    lenientFilter = ConeFilter(colorMatchThreshold=0.2)

    outputDir = os.path.join(os.path.dirname(__file__), '..', 'output', 'color_similarity_demo')
    os.makedirs(outputDir, exist_ok=True)

    initialDetection = [
        (10.0, 0.0, 0.7, 0.3, 0.0, 0.0),
    ]

    strictFilter.update(initialDetection, 0.0, 0.0, 0.0)
    lenientFilter.update(initialDetection, 0.0, 0.0, 0.0)

    strictFilter.visualize(savePath=os.path.join(outputDir, 'initial.png'), showPlot=False)
    print(f"Initial cone: blue=0.7, yellow=0.3")

    slightlyDifferent = [
        (10.0, 0.0, 0.5, 0.5, 0.0, 0.0),
    ]

    strictFilter.update(slightlyDifferent, 0.0, 0.0, 0.0)
    strictFilter.visualize(savePath=os.path.join(outputDir, 'strict_threshold.png'), showPlot=False)
    print(f"Strict threshold (0.7): {len(strictFilter.cones)} cones (no match)")

    lenientFilter.update(slightlyDifferent, 0.0, 0.0, 0.0)
    lenientFilter.visualize(savePath=os.path.join(outputDir, 'lenient_threshold.png'), showPlot=False)
    print(f"Lenient threshold (0.2): {len(lenientFilter.cones)} cones (matched)")

    print(f"\n✓ Saved color similarity demo to {outputDir}/")
    print("  Observe: threshold affects matching behavior")

if __name__ == '__main__':
    print("Generating Part 5 Color Confidence Visualizations")
    print("=" * 50)

    demonstrateColorConfidenceEvolution()
    demonstrateColorSimilarityMatching()

    print("\n" + "=" * 50)
    print("All Part 5 visualizations generated! ✓")
