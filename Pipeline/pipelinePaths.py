import os
import sys

# Adds the sibling module directories to sys.path so pipeline nodes can import
# the existing perception/localization/planning code the same way main.py does.
repoRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_paths = [
    os.path.join(repoRoot, 'Integration'),
    os.path.join(repoRoot, 'DepthEstimation'),
    os.path.join(repoRoot, 'Localization'),
    os.path.join(repoRoot, 'PathPlanning'),
    os.path.join(repoRoot, 'Segmentation', 'yolact_edge'),
    os.path.join(repoRoot, 'DepthEstimation', 'Depth-Anything-3', 'src'),
]

for _p in _paths:
    if _p not in sys.path:
        sys.path.append(_p)
