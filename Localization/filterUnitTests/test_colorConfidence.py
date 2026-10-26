import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from filter import ConeFilter

def testColorConfidenceNormalization():
    print("\n=== Test 1: Color Confidence Normalization ===")
    coneFilter = ConeFilter()

    detections = [
        (10.0, 0.0, 0.6, 0.3, 0.1, 0.0),
    ]
    coneFilter.update(detections, 0.0, 0.0, 0.0)

    totalConf = sum(coneFilter.cones[0].colorConfidences)
    print(f"Input: [0.6, 0.3, 0.1, 0.0] -> sum={0.6+0.3+0.1+0.0:.1f}")
    print(f"Normalized: {[f'{c:.3f}' for c in coneFilter.cones[0].colorConfidences]} -> sum={totalConf:.3f}")

    assert abs(totalConf - 1.0) < 0.001, "Confidences should sum to 1.0"

    print("✓ Color confidences normalized correctly")

def testColorConfidenceFusion():
    print("\n=== Test 2: Color Confidence Fusion ===")
    coneFilter = ConeFilter()

    detections = [
        (10.0, 0.0, 0.9, 0.1, 0.0, 0.0),
    ]
    coneFilter.update(detections, 0.0, 0.0, 0.0)

    print(f"Initial (obs=1): blue={coneFilter.cones[0].colorConfidences[0]:.3f}")

    slightlyDifferent = [
        (10.0, 0.0, 0.7, 0.3, 0.0, 0.0),
    ]
    coneFilter.update(slightlyDifferent, 0.0, 0.0, 0.0)

    print(f"After fusion (obs=2): blue={coneFilter.cones[0].colorConfidences[0]:.3f}, yellow={coneFilter.cones[0].colorConfidences[1]:.3f}")

    assert coneFilter.cones[0].colorConfidences[0] > 0.7, "Blue should still be dominant"
    assert coneFilter.cones[0].colorConfidences[0] < 0.9, "But should be influenced by new observation"
    assert coneFilter.cones[0].colorConfidences[1] > 0.1, "Yellow should increase"

    totalConf = sum(coneFilter.cones[0].colorConfidences)
    assert abs(totalConf - 1.0) < 0.001, "Confidences should still sum to 1.0"

    print("✓ Color confidences fused correctly")

def testColorConfidenceConvergence():
    print("\n=== Test 3: Color Confidence Convergence ===")
    coneFilter = ConeFilter()

    initialDetection = [
        (10.0, 0.0, 0.5, 0.5, 0.0, 0.0),
    ]
    coneFilter.update(initialDetection, 0.0, 0.0, 0.0)

    print(f"Initial (ambiguous): blue={coneFilter.cones[0].colorConfidences[0]:.3f}, yellow={coneFilter.cones[0].colorConfidences[1]:.3f}")

    for i in range(5):
        blueDetection = [
            (10.0, 0.0, 0.9, 0.1, 0.0, 0.0),
        ]
        coneFilter.update(blueDetection, 0.0, 0.0, 0.0)
        print(f"After obs {i+2}: blue={coneFilter.cones[0].colorConfidences[0]:.3f}, yellow={coneFilter.cones[0].colorConfidences[1]:.3f}")

    assert coneFilter.cones[0].colorConfidences[0] > 0.7, "Should converge to blue after repeated observations"

    print("✓ Color confidence converges over time")

def testDotProductSimilarity():
    print("\n=== Test 4: Dot Product Similarity ===")
    coneFilter = ConeFilter()

    conf1 = [0.9, 0.1, 0.0, 0.0]
    conf2 = [0.8, 0.2, 0.0, 0.0]
    conf3 = [0.0, 0.0, 0.9, 0.1]

    conf1Norm = coneFilter.normalizeColorConfidences(conf1)
    conf2Norm = coneFilter.normalizeColorConfidences(conf2)
    conf3Norm = coneFilter.normalizeColorConfidences(conf3)

    sim12 = coneFilter.colorConfidenceSimilarity(conf1Norm, conf2Norm)
    sim13 = coneFilter.colorConfidenceSimilarity(conf1Norm, conf3Norm)

    print(f"Blue vs Similar Blue: {sim12:.3f}")
    print(f"Blue vs Orange: {sim13:.3f}")

    assert sim12 > sim13, "Similar colors should have higher similarity"
    assert sim12 > 0.7, "Similar distributions should have high dot product"

    print("✓ Dot product similarity works correctly")

def testEnhancedColorMatching():
    print("\n=== Test 5: Enhanced Color Matching ===")
    coneFilter = ConeFilter(colorMatchThreshold=0.3)

    detections = [
        (10.0, 0.0, 0.7, 0.3, 0.0, 0.0),
    ]
    coneFilter.update(detections, 0.0, 0.0, 0.0)

    print(f"Initial cone: blue=0.7, yellow=0.3")

    similarDetection = [
        (10.0, 0.0, 0.6, 0.4, 0.0, 0.0),
    ]
    coneFilter.update(similarDetection, 0.0, 0.0, 0.0)

    print(f"Similar detection (blue=0.6, yellow=0.4): {len(coneFilter.cones)} cones")
    assert len(coneFilter.cones) == 1, "Similar color distributions should match"

    dissimilarDetection = [
        (10.0, 0.0, 0.0, 0.0, 0.9, 0.1),
    ]
    coneFilter.update(dissimilarDetection, 0.0, 0.0, 0.0)

    print(f"Dissimilar detection (orange=0.9): {len(coneFilter.cones)} cones")
    assert len(coneFilter.cones) == 2, "Dissimilar color distributions should not match"

    print("✓ Enhanced color matching works correctly")

def testMisclassificationCorrection():
    print("\n=== Test 6: Misclassification Correction ===")
    coneFilter = ConeFilter()

    slightlyWrongDetection = [
        (10.0, 0.0, 0.4, 0.6, 0.0, 0.0),
    ]
    coneFilter.update(slightlyWrongDetection, 0.0, 0.0, 0.0)

    print(f"Initial (slightly wrong): blue={coneFilter.cones[0].colorConfidences[0]:.3f}, yellow={coneFilter.cones[0].colorConfidences[1]:.3f}")

    for i in range(10):
        correctDetection = [
            (10.0, 0.0, 0.9, 0.1, 0.0, 0.0),
        ]
        coneFilter.update(correctDetection, 0.0, 0.0, 0.0)

    print(f"After 10 correct obs: blue={coneFilter.cones[0].colorConfidences[0]:.3f}, yellow={coneFilter.cones[0].colorConfidences[1]:.3f}")

    assert coneFilter.cones[0].colorConfidences[0] > coneFilter.cones[0].colorConfidences[1], "Should correct to blue"
    assert coneFilter.cones[0].colorConfidences[0] > 0.7, "Blue confidence should be strong"

    print("✓ Misclassification corrected over time")

def testOrangeAmbiguityResolution():
    print("\n=== Test 7: Orange Ambiguity Resolution ===")
    coneFilter = ConeFilter()

    ambiguousDetection = [
        (10.0, 0.0, 0.0, 0.0, 0.5, 0.5),
    ]
    coneFilter.update(ambiguousDetection, 0.0, 0.0, 0.0)

    print(f"Initial (ambiguous): sOrange={coneFilter.cones[0].colorConfidences[2]:.3f}, lOrange={coneFilter.cones[0].colorConfidences[3]:.3f}")

    for i in range(5):
        largeOrangeDetection = [
            (10.0, 0.0, 0.0, 0.0, 0.1, 0.9),
        ]
        coneFilter.update(largeOrangeDetection, 0.0, 0.0, 0.0)

    print(f"After resolution: sOrange={coneFilter.cones[0].colorConfidences[2]:.3f}, lOrange={coneFilter.cones[0].colorConfidences[3]:.3f}")

    assert coneFilter.cones[0].colorConfidences[3] > coneFilter.cones[0].colorConfidences[2], "Should resolve to large orange"

    print("✓ Orange ambiguity resolved correctly")

def testZeroConfidenceHandling():
    print("\n=== Test 8: Zero Confidence Handling ===")
    coneFilter = ConeFilter()

    zeroConfidences = [
        (10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    coneFilter.update(zeroConfidences, 0.0, 0.0, 0.0)

    print(f"Zero input normalized to: {[f'{c:.3f}' for c in coneFilter.cones[0].colorConfidences]}")

    totalConf = sum(coneFilter.cones[0].colorConfidences)
    assert abs(totalConf - 1.0) < 0.001, "Zero confidences should be normalized to uniform distribution"

    for conf in coneFilter.cones[0].colorConfidences:
        assert abs(conf - 0.25) < 0.001, "Each class should have equal probability"

    print("✓ Zero confidences handled correctly")

if __name__ == '__main__':
    print("Running Part 5 Color Confidence Tests")
    print("=" * 50)

    testColorConfidenceNormalization()
    testColorConfidenceFusion()
    testColorConfidenceConvergence()
    testDotProductSimilarity()
    testEnhancedColorMatching()
    testMisclassificationCorrection()
    testOrangeAmbiguityResolution()
    testZeroConfidenceHandling()

    print("\n" + "=" * 50)
    print("All Part 5 tests passed! ✓")
