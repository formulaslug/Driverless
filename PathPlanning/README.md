# Path Planning Module

This module implements real-time path planning for the UCSC FSAE driverless vehicle, generating optimal racing paths at 25 Hz.

**Based on**: AMZ Driverless (ETH Zurich) boundary estimation algorithm
- **Paper**: "AMZ Driverless: The Full Autonomous Racing System" (Kabzan et al., 2019)
- **Competition Results**: 1st place FSG 2017, 2018, FSI 2018
- **Performance**: 28.48s fastest lap, 1.5g lateral acceleration
- **Reference**: arXiv:1905.05150v1, Section 4.3
- **GitHub**: https://github.com/AMZ-Driverless

## Current Status (Dec 2025)

**Overall Progress**: 3 of 5 core phases complete (60%)

✅ **Completed**:
- Delaunay Triangulation with vectorized performance (0.8-2.5ms, 400-1250 Hz)
- Path Tree Population with breadth-first search and forward-arc filtering
- Path Smoothing with cubic splines and Formula Student constraint validation

⏳ **Remaining**:
- Cost Function implementation (5 AMZ terms)
- Beam Search pruning for path selection
- Control Planning interface definition

**Performance**: Currently well within 40ms budget. Delaunay + smoothing = ~5-12ms total.

## Module Overview

### Core Files

1. **config.py** - Configuration and constants
   - Track constraints (min width 3m, turning radius 4.5m)
   - Performance parameters (25 Hz, beam width)
   - Cost function weights (qc, qw, qs, qcolor, ql)
   - Vehicle dynamics parameters
   - Sensor range (~10m)

2. **delaunay.py** - Delaunay Triangulation ✅ **[COMPLETE]**
   - Uses `scipy.spatial.Delaunay` for triangulation (NOT CGAL)
   - Creates triangulation from cone [x, y] positions
   - Extracts midpoint graph for waypoints with adjacency connections
   - **Important**: Handles duplicate midpoints (interior edges shared by 2 triangles)
   - Vectorized implementation for performance (>100 Hz on test data)
   - Multi-track visualization utilities (matplotlib)
   - **AMZ Implementation**: Used CGAL library, we use scipy for simplicity
   - **Performance**: 0.8-2.5ms for 10-30 cones (400-1250 Hz)
   - **Status**: ✅ Complete

3. **path_tree.py** - Path Tree Population (Breadth-First) ✅ **[COMPLETE]**
   - Builds breadth-first tree of possible paths through midpoint graph
   - `find_nearest_waypoints()`: Selects k nearest waypoints in forward arc (±90°)
   - `get_path_tree()`: Grows tree from vehicle position with configurable depth
   - Prevents loops by tracking visited waypoints in each path
   - Forward-only constraint (no backwards waypoints)
   - Limited tree depth to meet 40ms cycle time constraint
   - Rich visualization showing paths, cones, vehicle, and heading
   - **AMZ approach**: Breadth-first manner, fixed iteration limit
   - **Status**: ✅ Complete

4. **cost_functions.py** - Path Cost Evaluation (5 AMZ Terms)

   Implements the exact AMZ cost function with 5 weighted terms:

   - **Term 1**: Maximum angle change between path segments
     - Rationale: Sharp corners use multiple cones, sudden angles unlikely

   - **Term 2**: Track width standard deviation (~3m expected)
     - Rationale: Rules specify 3m min width, unlikely to change drastically

   - **Term 3**: Cone spacing standard deviation (~5m max expected)
     - Rationale: Cones typically spaced evenly along boundaries

   - **Term 4**: Wrong color probability (blue=left, yellow=right)
     - Rationale: Uses probabilistic color estimates from perception
     - Becomes zero if no color information available

   - **Term 5**: Path length deviation from sensor range (~10m)
     - Rationale: Penalize too short/long paths, prefer sensor range length

   **Cost Function Formula**:
   ```python
   cost = qc * normalize(max_angle_change)² +
          qw * normalize(track_width_stddev)² +
          qs * normalize(cone_spacing_stddev)² +
          qcolor * normalize(wrong_color_prob)² +
          ql * normalize(path_length_diff)²
   ```

   - Each term normalized, squared, and weighted
   - Weights (qc, qw, qs, qcolor, ql) tuned experimentally
   - **Status**: Not Started (Deadline: Dec 3)

5. **beam_search.py** - Beam Search Pruning
   - Prunes path tree to keep only top-k best paths at each level
   - Prevents exponential growth of search tree
   - Limits computational complexity for real-time performance
   - Integrates cost function evaluation
   - **AMZ result**: Only 4.2% of paths left track, mostly at >7m distance
   - **Status**: Not Started (Deadline: Nov 29)

6. **path_smoother.py** - Spline Interpolation ✅ **[COMPLETE]**
   - `smooth_path()`: Smooths discrete waypoints using cubic splines (splprep/splev)
   - Calculates curvature at each point: κ = (dx·d²y - dy·d²x) / (dx² + dy²)^1.5
   - Validates against Formula Student constraints (min 9m turning diameter)
   - Multi-panel visualization: path with curvature coloring, curvature profile, turn radius
   - Pass/fail indicators for each test track
   - Test suite includes 6 track configurations (straight, curve, hairpin, slalom, oval, s-curve)
   - **AMZ approach**: Feeds smoothed path to pure pursuit controller
   - **Status**: ✅ Complete

7. **path_planner.py** - Main Integration Module
   - 25 Hz planning loop (40ms cycle time)
   - Integrates all 5 phases of algorithm
   - SLAM interface (input): receives cone positions at 25 Hz
   - Control planning interface (output): sends path data at 25 Hz
   - Real-time performance monitoring
   - **AMZ timing**: 50ms in SLAM mode, 40ms in racing mode
   - **Status**: Not Started (Deadline: Dec 6)

## Algorithm Pipeline

Based on AMZ Section 4.3: Boundary Estimation

```
SLAM -> Cones -> Delaunay -> Waypoints -> Path Tree -> Beam Search -> Best Path -> Spline Smooth -> Control
  |                                                                                                    |
  +------------------------------------ 25 Hz Loop (40ms cycle time) ----------------------------------+
```

### Phase 1: Graph Generation (Delaunay Triangulation)
- **Input**: Cone positions [x, y] from SLAM
- **Library**: scipy.spatial.Delaunay (AMZ uses CGAL)
- **Process**:
  1. Create Delaunay triangulation of all visible cones
  2. Triangles maximize minimum internal angles
  3. Extract midpoints of all triangle edges
  4. Handle duplicate midpoints from shared interior edges
- **Output**: Midpoint waypoint graph
- **AMZ approach**: Discretizes X-Y space into connected triangles

### Phase 2: Tree Population (Breadth-First Search)
- **Input**: Waypoints + current vehicle position
- **Process**:
  1. Start from vehicle position
  2. Build tree by connecting to surrounding edge midpoints
  3. Each level represents one waypoint ahead
  4. Branches represent alternative route choices
  5. Limit depth by iteration count (computational budget)
- **Output**: Tree of all possible paths
- **AMZ approach**: Breadth-first manor, explores all possible center lines

### Phase 3: Beam Search (Cost-Based Pruning)
- **Input**: Path tree from Phase 2
- **Process**:
  1. Evaluate each path using 5-term cost function
  2. Normalize, square, and weight each cost term
  3. Keep only top-k paths with lowest cost at each level
  4. Discard worst paths (beam search pruning)
- **Output**: Top-k best paths
- **AMZ result**: 4.2% invalid paths at FSG 2018, mostly >7m away

### Phase 4: Path Selection & Smoothing
- **Input**: Best path waypoints from Phase 3
- **Process**:
  1. Select path with absolute lowest cost
  2. Fit spline through discrete waypoints
  3. Calculate curvature along path
- **Output**: Smooth continuous path with curvature
- **AMZ approach**: Passes to pure pursuit controller

### Phase 5: Control Interface
- **Input**: Smooth path + curvatures
- **Process**:
  1. Calculate velocity profile (optional, may be control team's job)
  2. Package waypoints, velocities, curvatures
  3. Send to control planning at 25 Hz
- **Output**: Path data for control planning
- **Coordinate with**: Ray Zou, Nathan Margolis (Control Planning team)

## Running Tests

Each module has standalone tests in its `if __name__ == "__main__"` block:

```bash
# Test Delaunay triangulation ✅
python PathPlanning/delaunay.py

# Test path tree ✅
python PathPlanning/path_tree.py

# Test path smoother ✅
python PathPlanning/path_smoother.py

# Test cost functions (not yet implemented)
python PathPlanning/cost_functions.py

# Test beam search (not yet implemented)
python PathPlanning/beam_search.py

# Test full integrated system (not yet implemented)
python PathPlanning/path_planner.py
```

### Test Outputs

**delaunay.py**:
- Prints performance benchmarks for 5 test tracks (cones, time, Hz)
- Shows multi-panel visualization of triangulation + waypoint graph
- Validates vectorized implementation performance

**path_tree.py**:
- Generates all possible paths from vehicle position through waypoint graph
- Visualizes up to 10 sample paths with different colors/styles
- Shows vehicle position, heading arrow, cones, and available waypoints
- Forward-arc filtering demonstration (±90° from vehicle heading)

**path_smoother.py**:
- Generates 6 test tracks (straight, simple curve, hairpin, slalom, oval, s-curve)
- Shows 3-panel visualization for each track:
  - Path with curvature color coding (red = high curvature)
  - Curvature profile vs. path progress
  - Turn radius with Formula Student compliance zones
- Prints pass/fail status for min 9m turning diameter constraint

## Performance Requirements

- **Frequency**: 25 Hz (40ms cycle time)
- **Hardware**: Raspberry Pi (limited CPU/memory)
- **Real-time**: Must complete before next SLAM update
- **Robustness**: Handle noisy monocular vision data
- **AMZ benchmark**: 50ms cycle time in SLAM mode, achieved 1.5g lateral acceleration

### Computational Budget (Current Performance)

- Delaunay triangulation: ✅ 0.8-2.5ms for 10-30 cones (400-1250 Hz)
- Tree building: TBD (depends on max_depth and k_start parameters)
- Cost evaluation: ⏳ Not yet implemented
- Beam search: ⏳ Not yet implemented
- Spline smoothing: ✅ ~5-10ms (acceptable for 25 Hz)
- **Total target**: <40ms on Raspberry Pi
- **Current progress**: 3/5 phases complete, well within budget so far

## Key Implementation Differences from AMZ

| Aspect | AMZ (ETH Zurich) | Our Implementation |
|--------|------------------|-------------------|
| Triangulation Library | CGAL (C++) | scipy.spatial.Delaunay (Python) |
| Update Frequency | 50ms (20 Hz) SLAM mode | 40ms (25 Hz) target |
| Hardware | Industrial PC + GPU | Raspberry Pi (CPU only) |
| Sensors | LiDAR + Stereo Camera | Monocular Camera only |
| Programming Language | C++ / ROS | Python / ROS |
| Sensor Range | ~10m (LiDAR) | ~10m (monocular depth estimation) |

**Key Challenge**: Monocular vision provides noisier/sparser cone detections than LiDAR, so our cost function needs to be more robust.

## Integration Points

### Input from Localization/SLAM
- **Team**: Parker Costa, Hansika Nerusa, Chanchal Mukeshsingh
- **Format**: TBD - coordinate on cone position data structure
  - Need: [x, y] positions in global frame
  - Need: Color probabilities [p_blue, p_yellow, p_orange, p_unknown]
  - Need: Position uncertainties (covariance matrices)
- **Update rate**: 25 Hz

### Output to Control Planning
- **Team**: Ray Zou, Nathan Margolis
- **Format**: TBD - coordinate on path/velocity data structure
  - Provide: Waypoint coordinates [x, y] array
  - Provide: Desired velocities at each waypoint (optional)
  - Provide: Curvature at each waypoint
  - Provide: Path confidence/quality metric
- **Update rate**: 25 Hz

## Development Timeline

| Task | Deadline | Status |
|------|----------|--------|
| Delaunay Triangulation (scipy) | Nov 8 | ✅ **COMPLETE** |
| Test & Visualize | Nov 15 | ✅ **COMPLETE** |
| Path Tree Population | Nov 22 | ✅ **COMPLETE** |
| Path Smoothing (scipy splines) | Dec 3 | ✅ **COMPLETE** |
| Beam Search Pruning | Nov 29 | ⏳ **NOT STARTED** |
| Cost Functions (5 AMZ terms) | Dec 3 | ⏳ **NOT STARTED** |
| Control Interface | Dec 6 | ⏳ **NOT STARTED** |

## Dependencies

```python
import numpy as np
from scipy.spatial import Delaunay          # Delaunay triangulation
from scipy.interpolate import splprep, splev, CubicSpline  # Path smoothing
import matplotlib.pyplot as plt              # Visualization
```

No CGAL required! Everything uses standard scipy/numpy.

## AMZ Cost Function Tuning Guide

The 5 cost weights need experimental tuning on your specific tracks:

```python
# In config.py - starting values (tune these!)
qc = 1.0      # Angle change weight
qw = 1.0      # Track width variance weight
qs = 1.0      # Cone spacing variance weight
qcolor = 2.0  # Color mismatch weight (higher = trust colors more)
ql = 0.5      # Path length deviation weight
```

**Tuning strategy**:
1. Start with all weights = 1.0
2. Test on simple_track and oval_track patterns
3. Visualize which paths are selected
4. Increase weights for terms that should be more important
5. qcolor should be high if perception gives good colors, low if noisy
6. AMZ used squared and normalized costs, so relative weights matter more than absolute

## Team

- Nathan Yee
- Suhani Agarwal
- Aedan Benavides

## References

**Primary**:
- Kabzan et al. "AMZ Driverless: The Full Autonomous Racing System", 2019
- Section 4.3: Boundary Estimation (our algorithm)
- Section 5: MPC Control (Control Planning team's reference)

**Additional**:
- Delaunay, B. "Sur la sphère vide", 1934 (original Delaunay paper)
- scipy.spatial.Delaunay documentation
- Formula Student Germany Rules (track specifications)
