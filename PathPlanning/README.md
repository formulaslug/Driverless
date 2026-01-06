# Path Planning Module

Real-time path planning for UCSC FSAE driverless racing at **25 Hz (40ms cycle time)**

**Based on**: AMZ Driverless (ETH Zurich) - 1st place FSG 2017/2018 (arXiv:1905.05150v1, Section 4.3)
**Team**: Nathan Yee, Suhani Agarwal, Aedan Benavides

## Status (Jan 2026)

**Progress**: 3/5 phases complete (60%) | Performance: Well within 40ms budget

| Phase | Status | File | Performance |
|-------|--------|------|-------------|
| 1. Delaunay Triangulation | ✅ COMPLETE | `delaunay.py` | 0.8-2.5ms (400-1250 Hz) |
| 2. Path Tree Population | ✅ COMPLETE | `path_tree.py` | TBD |
| 3. Cost Function + Beam Search | ⏳ NOT STARTED | `cost_functions.py`, `beam_search.py` | - |
| 4. Path Smoothing | ✅ COMPLETE | `path_smoother.py` | ~5-10ms |
| 5. Main Integration | 🚧 IN PROGRESS | `path_planner.py` | - |

**Current Total**: ~5-12ms (Delaunay + smoothing only) → 28ms remaining for cost/beam search

## Algorithm Pipeline

```
SLAM (cones) → Delaunay → Waypoints → Path Tree → Beam Search → Best Path → Smooth → Control
                └─────────────────── 25 Hz (40ms cycle) ──────────────────────┘
```

### Files & Responsibilities

| File | Status | Key Functions | Notes |
|------|--------|---------------|-------|
| `config.py` | ✅ | Constants & parameters | Track constraints, weights, beam width, vehicle params |
| `delaunay.py` | ✅ | `get_midpoints(cones)` | Returns waypoints, graph, triangulation. 0.8-2.5ms. |
| `path_tree.py` | ✅ | `get_path_tree(cones, vehicle_pos, heading, depth, k)` | Breadth-first tree with forward-arc filtering (±90°). Prevents cycles. |
| `cost_functions.py` | ⏳ | `evaluate_path_cost(path, cones, colors)` | **NEEDED**: 5-term AMZ cost function (angle, width, spacing, color, length) |
| `beam_search.py` | ⏳ | `beam_search_prune(tree, beam_width, cost_fn)` | **NEEDED**: Keep top-k paths per level |
| `path_smoother.py` | ✅ | `smooth_path(waypoints, num_points)` | Cubic splines + curvature. Returns (path, curvature). ~5-10ms. |
| `path_planner.py` | 🚧 | `plan_path(cones, colors, vehicle_pos, heading)` | **NEEDED**: End-to-end integration of all phases |

### 5-Term AMZ Cost Function (TO IMPLEMENT)

```python
cost = qc·norm(max_angle_change)² + qw·norm(track_width_stddev)² +
       qs·norm(cone_spacing_stddev)² + qcolor·norm(wrong_color_prob)² +
       ql·norm(path_length_diff)²
```

| Term | Expected Value | Rationale |
|------|----------------|-----------|
| Max Angle Change | Penalize sharp turns | Corners use multiple cones, sudden angles unlikely |
| Track Width StdDev | ~3m consistent | Rules: 3m min width, unlikely to vary |
| Cone Spacing StdDev | ~5m max | Cones evenly spaced along boundaries |
| Wrong Color Prob | Blue=left, Yellow=right | Probabilistic color from perception |
| Path Length Deviation | ~10m target | Prefer paths matching sensor range |

**Weights** (in `config.py`, need tuning): `qc`, `qw`, `qs`, `qcolor`, `ql`

## Testing

**Status**: Tests have been removed from module files. Separate test files will be created.

**Planned test structure**:
```bash
PathPlanning/tests/
  ├── test_delaunay.py           # ⏳ To be created - Benchmarks, visualization
  ├── test_path_tree.py          # ⏳ To be created - Path generation, forward-arc filtering
  ├── test_path_smoother.py      # ⏳ To be created - 6 test tracks, curvature validation
  ├── test_cost_functions.py     # ⏳ To be created - Cost function evaluation
  ├── test_beam_search.py        # ⏳ To be created - Pruning algorithm
  └── test_path_planner.py       # ⏳ To be created - End-to-end integration
```

**Test coverage needed**:
- Delaunay: Performance benchmarks (0.8-2.5ms), multi-panel visualization
- Path Tree: Path generation, vehicle heading, forward-arc filtering (±90°)
- Path Smoother: 6 test tracks (straight, curve, hairpin, slalom, oval, s-curve), Formula Student compliance
- Cost Functions: Each of 5 terms, normalization, edge cases
- Beam Search: Pruning logic, beam width parameter
- Path Planner: Full pipeline, 40ms performance requirement

## Performance Budget (40ms total on Raspberry Pi)

| Component | Current | Status | Notes |
|-----------|---------|--------|-------|
| Delaunay | 0.8-2.5ms | ✅ | 10-30 cones, vectorized scipy |
| Path Tree | TBD | ✅ | Depends on `max_depth`, `k_start` params |
| Cost Function | - | ⏳ | Not implemented |
| Beam Search | - | ⏳ | Not implemented |
| Spline Smooth | ~5-10ms | ✅ | Cubic splines, acceptable |
| **Total** | **~5-12ms** | **3/5 phases** | **28ms remaining** |

**Key Challenge**: Monocular camera (vs AMZ's LiDAR) → noisier cone detections → cost function must be robust

## Integration (Coordinate with Other Teams)

**Input from SLAM** (Parker Costa, Hansika Nerusa, Chanchal Mukeshsingh) @ 25 Hz:
- Cone positions [x, y] in global frame
- Color probabilities [p_blue, p_yellow, p_orange, p_unknown]
- Position uncertainties (optional: covariance matrices)

**Output to Control** (Ray Zou, Nathan Margolis) @ 25 Hz:
- Waypoint coordinates [x, y] array
- Curvature at each waypoint
- Desired velocities (optional, may be control's responsibility)
- Path confidence/quality metric

## Next Steps (Priority Order)

1. **Implement `cost_functions.py`** (HIGHEST PRIORITY)
   - 5-term AMZ cost function
   - Normalization for each term
   - Handle edge cases (missing cones, sparse data)

2. **Implement `beam_search.py`**
   - Keep top-k paths per tree level
   - Use `config.BEAM_WIDTH` parameter

3. **Complete `path_planner.py`**
   - Fix incomplete imports (lines 4-5)
   - Integrate all 5 phases
   - Add performance profiling (must hit 40ms)

4. **Create Test Suite**
   - Set up `PathPlanning/tests/` directory
   - Create test files for each module (see Testing section above)
   - Port existing test patterns (oval, slalom, grid) to separate test files

5. **Test & Validate**
   - Thread cone colors through pipeline
   - Test on all track patterns
   - Profile on Raspberry Pi

## Cost Function Tuning Guide

Start with equal weights, then tune based on visualization:

```python
# In config.py - starting values
qc = 1.0      # Angle change
qw = 1.0      # Track width variance
qs = 1.0      # Cone spacing variance
qcolor = 2.0  # Color mismatch (increase if colors are reliable)
ql = 0.5      # Path length deviation
```

**Strategy**: Test on simple/oval tracks → visualize selected paths → adjust weights → repeat

## Dependencies

```python
import numpy as np
from scipy.spatial import Delaunay                    # Delaunay triangulation
from scipy.interpolate import splprep, splev          # Path smoothing
import matplotlib.pyplot as plt                       # Visualization
```

**No CGAL needed** - scipy handles everything!
