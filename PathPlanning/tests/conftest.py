import sys
import os

# Add PathPlanning/ so tests import the production source modules (config, delaunay, ...)
# ahead of the helper modules in tests/ (track_generator, utils).
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_tests_dir)
sys.path.insert(0, _tests_dir)
sys.path.insert(0, _src_dir)

# Non-interactive backend so plt.show() in the demo plots never blocks a test run.
import matplotlib
matplotlib.use('Agg')
