import os
import sys

# Make ControlPlanning/ importable (controller, control_config, lateral/, longitudinal/)
CONTROL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CONTROL_DIR not in sys.path:
    sys.path.insert(0, CONTROL_DIR)

