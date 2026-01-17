# Path Planning Module

Real-time path planning for UCSC FSAE driverless racing at **25 Hz (40ms cycle time)**

**Based on**: AMZ Driverless (ETH Zurich) methodology - arXiv:1905.05150v1
**Team**: Nathan Yee, Suhani Agarwal, Aedan Benavides

## Status (Jan 2026)

**Progress**: 4/5 phases complete (80%) | Performance: Well within 40ms budget

| Phase | Status | File |
|-------|--------|------|
| 1. Delaunay Triangulation | ✅ COMPLETE | `delaunay.py` |
| 2. Path Tree Population | ✅ COMPLETE | `path_tree.py` |
| 3. Cost Functions | ✅ COMPLETE | `cost_functions.py` |
| 4. Path Smoothing | ✅ COMPLETE | `path_smoother.py` |
| 5. Beam Search + Integration | 🚧 IN PROGRESS | `beam_search.py`, `path_planner.py` |

## How It Works

```
SLAM → Delaunay Triangulation → Path Tree → Cost Evaluation → Beam Search → Smoothing → Control
       └──────────────────────────── 25 Hz (40ms cycle) ────────────────────────────┘
```

### Core Modules

| Module | Function | Description |
|--------|----------|-------------|
| `delaunay.py` | `get_midpoints(cones, colors)` | Generates waypoint graph using Delaunay triangulation. Only creates waypoints between blue-yellow cone pairs (track boundaries). |
| `path_tree.py` | `get_path_tree(cones, colors, vehicle_pos, heading, depth, k)` | Breadth-first search through waypoint graph. Forward-arc filtering (±90°) and cycle prevention. |
| `cost_functions.py` | `evaluate_path_cost(path, cones, colors)` | 7-metric AMZ cost function: angle sharpness, smoothness, spacing, width variance, color confidence, length, boundary violations. |
| `path_smoother.py` | `smooth_path(waypoints, num_points)` | Cubic spline interpolation. Returns smooth coordinates + curvature array. |
| `beam_search.py` | `beam_search_prune(tree, beam_width, cost_fn)` | **TODO**: Prune path tree to top-k lowest cost paths. |
| `path_planner.py` | `plan_path(cones, colors, vehicle_pos, heading)` | **TODO**: End-to-end integration of all phases. |

## Testing

```bash
PathPlanning/tests/
  ├── test_delaunay.py           # ✅ Visualization for straight/curved/hairpin/chicane tracks
  ├── test_path_tree.py          # ✅ Path generation, forward-arc filtering
  ├── test_cost_functions.py     # ✅ Cost evaluation on realistic tracks (straight/curved/hairpin/slalom/spare cones)
  ├── test_beam_search.py        # ⏳ TODO - Pruning algorithm
  └── test_path_planner.py       # ⏳ TODO - End-to-end integration
```

Run tests: `cd tests && python test_<module>.py`

## Data Interfaces

**Input from SLAM** @ 25 Hz:
- `cones`: np.ndarray (n_cones, 2) - [x, y] positions in vehicle frame
- `colors`: np.ndarray (n_cones, 4) - [p_blue, p_yellow, p_orange_small, p_orange_large]

**Output to Control** @ 25 Hz:
- Smooth waypoint path: np.ndarray (n_points, 2)
- Curvature array: np.ndarray (n_points,)
- Path quality/confidence metric

## Next Steps

1. **Implement `beam_search.py`** - Prune path tree to top-k paths per level
2. **Complete `path_planner.py`** - Integrate all phases into single API
3. **Performance profiling** - Must achieve <40ms on Raspberry Pi
4. **Cost function tuning** - Adjust weights in `config.py` based on real track data

## Dependencies

- NumPy - Array operations
- SciPy - Delaunay triangulation (`scipy.spatial`) and spline smoothing (`scipy.interpolate`)
- Matplotlib - Visualization (testing only)

Install: `conda activate fsae` (all dependencies already installed)
