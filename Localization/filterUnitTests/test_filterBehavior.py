import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from filter import ConeFilter

blue = (10.0, 0.0, 0.7, 0.3, 0.0, 0.0)
blueYellowSplit = (10.0, 0.0, 0.5, 0.5, 0.0, 0.0)

def testColorThresholdCanReject():
    strictFilter = ConeFilter(colorMatchThreshold=0.7)
    lenientFilter = ConeFilter(colorMatchThreshold=0.2)
    for coneFilter in (strictFilter, lenientFilter):
        coneFilter.update([blue], 0.0, 0.0, 0.0)
        coneFilter.update([blueYellowSplit], 0.0, 0.0, 0.0)

    assert len(strictFilter.cones) == 2, "Strict threshold should reject the dissimilar colors"
    assert len(lenientFilter.cones) == 1, "Lenient threshold should accept them"

def testUnnormalizedDetectionColorsMatch():
    coneFilter = ConeFilter()
    coneFilter.update([(10.0, 0.0, 9.0, 0.0, 0.0, 0.0)], 0.0, 0.0, 0.0)
    coneFilter.update([(10.0, 0.0, 0.9, 0.0, 0.0, 0.0)], 0.0, 0.0, 0.0)
    assert len(coneFilter.cones) == 1

def testStationaryCarAddsNoProcessNoise():
    coneFilter = ConeFilter()
    coneFilter.update([blue], 0.0, 0.0, 0.0)
    before = coneFilter.cones[0].cov.copy()
    coneFilter.update([], 0.0, 0.0, 0.0)
    assert np.allclose(coneFilter.cones[0].cov, before)

def testMotionAddsProcessNoise():
    coneFilter = ConeFilter()
    coneFilter.update([blue], 0.0, 0.0, 0.0)
    before = np.trace(coneFilter.cones[0].cov)
    coneFilter.update([], 1.0, 0.0, 0.05)
    assert np.trace(coneFilter.cones[0].cov) > before

def testConesBehindCarArePruned():
    coneFilter = ConeFilter(behindCarMargin=5.0)
    coneFilter.update([blue], 0.0, 0.0, 0.0)
    coneFilter.update([], 16.0, 0.0, 0.0)
    assert len(coneFilter.cones) == 0

def testCovarianceStaysSymmetric():
    coneFilter = ConeFilter()
    for i in range(20):
        coneFilter.update([(10.0 + 0.1 * (i % 3), 1.0, 0.9, 0.0, 0.0, 0.0)], 0.2, 0.05, 0.01)
    cov = coneFilter.cones[0].cov
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
