import sys
import os

# Add PathPlanning/ so tests can import source modules (delaunay, main, etc.)
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_tests_dir)
sys.path.insert(0, _src_dir)    # PathPlanning/
sys.path.insert(0, _tests_dir)  # PathPlanning/tests/ (for track_generator, utils)
