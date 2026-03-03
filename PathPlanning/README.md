# Path Planning Module — Integration Guide

## API

```python
from path_planner import plan_path

smooth_path, curvature = plan_path(cones, coordinate_confidence, colors, vehicle_pos, vehicle_heading)
```

Returns `(None, None)` if no valid path is found.

## Input (from SLAM)

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `cones` | `np.ndarray` | `(n, 2)` | Cone positions `[x, y]` in vehicle frame (meters) |
| `coordinate_confidence` | `np.ndarray` | `(n,)` | Positional uncertainty radius per cone (meters). `0.0` = perfect confidence. |
| `colors` | `np.ndarray` | `(n, 4)` | Color probabilities `[blue, yellow, orange_small, orange_large]` per cone. Each row sums to ~1.0. |
| `vehicle_pos` | `np.ndarray` | `(2,)` | Vehicle position `[x, y]` in the same frame as cones |
| `vehicle_heading` | `float` | scalar | Vehicle heading in **radians** (0 = +x direction) |

## Output (to Control)

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `smooth_path` | `np.ndarray` | `(100, 2)` | Smoothed waypoints `[x, y]` via cubic spline |
| `curvature` | `np.ndarray` | `(100,)` | Curvature at each waypoint (1/radius) |

Output point count is configurable via `SPLINE_NUM_POINTS` in `config.py`.

## Requirements

```
numpy
scipy
```

Matplotlib is only needed for running tests with visualization.

Environment: `conda activate fsae` (Python 3.12.2)

## Running Tests

```bash
cd PathPlanning
pytest tests/ -v        # all tests
pytest tests/ -v -s     # with matplotlib visualizations
```

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BEAM_WIDTH` | 10 | Paths kept per tree level during search |
| `MAX_TREE_DEPTH` | 30 | BFS depth limit |
| `K_START` | 3 | Starting waypoints at tree root |
| `SPLINE_NUM_POINTS` | 100 | Points in smoothed output path |
| `MIN_CONES_FOR_VALID_PATH` | 4 | Minimum cones needed to plan |

