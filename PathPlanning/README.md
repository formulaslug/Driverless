# Path Planning Module

Real-time path planning for UCSC FSAE driverless racing at **25 Hz (40ms cycle time)**

**Based on**: AMZ Driverless (ETH Zurich) methodology - arXiv:1905.05150v1
**Team**: Nathan Yee, Suhani Agarwal, Aedan Benavides

## Status (Jan 2026)

**✅ COMPLETE** - All 5 phases implemented and integrated

| Phase | Status | File |
|-------|--------|------|
| 1. Delaunay Triangulation | ✅ COMPLETE | `delaunay.py` |
| 2. Path Tree Population | ✅ COMPLETE | `path_tree.py` |
| 3. Cost Functions | ✅ COMPLETE | `cost_functions.py` |
| 4. Path Smoothing | ✅ COMPLETE | `path_smoother.py` |
| 5. Beam Search + Integration | ✅ COMPLETE | `beam_search.py`, `main.py` |

## How It Works

```
SLAM → Delaunay Triangulation → Path Tree → Cost Evaluation → Beam Search → Smoothing → Control
       └──────────────────────────── 25 Hz (40ms cycle) ────────────────────────────┘
```

### Core Modules

| Module | Function | Description |
|--------|----------|-------------|
| `delaunay.py` | `get_midpoints(cones, colors)` | Generates waypoint graph using Delaunay triangulation. Only creates waypoints between blue-yellow cone pairs (track boundaries). |
| `path_tree.py` | `get_path_tree(cones, colors, vehicle_pos, heading, depth, k)` | Breadth-first search through waypoint graph. Forward-arc filtering (±90°), cycle prevention, and level-by-level beam search pruning. |
| `cost_functions.py` | `evaluate_path_cost(path, cones, colors)` | 7-metric AMZ cost function: angle sharpness, smoothness, spacing, width variance, color confidence, length, boundary violations. |
| `path_smoother.py` | `smooth_path(waypoints, num_points)` | Cubic spline interpolation. Returns smooth coordinates + curvature array. |
| `beam_search.py` | `beam_search_prune(paths, cones, colors, beam_width)` | Prune path list to top-k lowest cost paths. |
| `main.py` | `plan_path(cones, colors, vehicle_pos, heading)` | **Main API** - End-to-end pipeline integrating all phases. Returns smooth path + curvature. |

## Testing

```bash
PathPlanning/tests/
  ├── test_delaunay.py           # ✅ Delaunay triangulation with visualization
  ├── test_path_tree.py          # ✅ Path tree generation, forward-arc filtering
  ├── test_cost_functions.py     # ✅ Cost evaluation on realistic tracks
  └── test_main.py               # ✅ End-to-end pipeline with full visualizations
```

Run tests: `cd tests && python test_main.py`

## Data Interfaces

**Input from SLAM** @ 25 Hz:
- `cones`: np.ndarray (n_cones, 2) - [x, y] positions in vehicle frame
- `colors`: np.ndarray (n_cones, 4) - [p_blue, p_yellow, p_orange_small, p_orange_large]

**Output to Control** @ 25 Hz:
- Smooth waypoint path: np.ndarray (n_points, 2)
- Curvature array: np.ndarray (n_points,)
- Path quality/confidence metric

## Usage

```python
from main import plan_path
import numpy as np

# Input from SLAM (cone positions and color probabilities)
cones = np.array([[5, 2], [10, 2], [15, 2], ...])
colors = np.array([[1, 0, 0, 0], [0, 1, 0, 0], ...])  # blue, yellow, orange_small, orange_large

# Vehicle state
vehicle_pos = np.array([0, 0])
vehicle_heading = 0.0  # radians

# Run path planner
smooth_path, curvature = plan_path(cones, colors, vehicle_pos, vehicle_heading)

if smooth_path is not None:
    # Send to Control Planning team
    print(f"Path shape: {smooth_path.shape}")  # (100, 2)
    print(f"Curvature shape: {curvature.shape}")  # (100,)
else:
    print("No valid path found")
```

## Next Steps

1. **Performance profiling** - Measure execution time on Raspberry Pi (target: <40ms)
2. **Real-world testing** - Test with actual SLAM cone detections
3. **Cost function tuning** - Adjust weights in `config.py` based on track data
4. **Integration** - Connect to SLAM and Control Planning modules

## Dependencies

- NumPy - Array operations
- SciPy - Delaunay triangulation (`scipy.spatial`) and spline smoothing (`scipy.interpolate`)
- Matplotlib - Visualization (testing only)

Install: `conda activate fsae` (all dependencies already installed)
