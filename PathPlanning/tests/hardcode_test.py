import os
import sys
import numpy as np
import pytest

# Ensure imports work when running from PathPlanning/tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from path_tree import get_path_tree
from beam_search import beam_search_prune
from cost_functions import evaluate_path_cost

# Headless plotting (no GUI needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# Hardcoded track layouts

def _colors_for(cones, blue_mask):
    """
    cones: (N,2)
    blue_mask: boolean array length N (True -> blue, False -> yellow)
    returns colors: (N,4)
    """
    colors = np.zeros((len(cones), 4), dtype=float)
    colors[blue_mask, cfg.CONE_COLOR_BLUE] = 0.95
    colors[~blue_mask, cfg.CONE_COLOR_YELLOW] = 0.95
    # small leftover probability mass (optional)
    colors[:, 2] = 0.02
    colors[:, 3] = 0.01
    # renorm (optional)
    colors = colors / np.sum(colors, axis=1, keepdims=True)
    return colors


def make_track_straight_zigzag(track_width=3.0):
    """
    Straight-ish track but with slight alternating y offsets so Delaunay
    triangles aren't too degenerate and your midpoint graph actually connects.
    """
    xs = np.linspace(0, 18, 10)
    cones = []
    blue_mask = []

    for i, x in enumerate(xs):
        # tiny "zigzag" to avoid all triangles having only 1 valid edge
        jitter = 0.35 if (i % 2 == 0) else -0.35

        # blue boundary
        cones.append([x,  track_width/2 + jitter])
        blue_mask.append(True)

        # yellow boundary
        cones.append([x, -track_width/2 - jitter])
        blue_mask.append(False)

    cones = np.array(cones, dtype=float)
    colors = _colors_for(cones, np.array(blue_mask, dtype=bool))
    return cones, colors


def make_track_gentle_s(track_width=3.0):
    """
    Gentle S-curve centerline; boundaries offset by +/- track_width/2.
    Uses sin to keep it smooth but still 2D enough for triangulation.
    """
    xs = np.linspace(0, 22, 12)
    cones = []
    blue_mask = []

    for i, x in enumerate(xs):
        center_y = 1.2 * np.sin(x / 4.0)
        # slight asymmetric jitter (helps graph connectivity)
        jitter = 0.25 * np.sin(1.7 * i)

        cones.append([x, center_y + track_width/2 + jitter])   # blue
        blue_mask.append(True)

        cones.append([x, center_y - track_width/2 - jitter])   # yellow
        blue_mask.append(False)

    cones = np.array(cones, dtype=float)
    colors = _colors_for(cones, np.array(blue_mask, dtype=bool))
    return cones, colors


def make_track_hairpin(track_width=3.0):
    """
    A U-ish turn made from two legs + a rounded-ish top.
    Still hardcoded cones; enough spread for Delaunay.
    """
    cones = []
    blue_mask = []

    # Leg 1 (up in x)
    xs1 = np.linspace(0, 10, 7)
    for i, x in enumerate(xs1):
        y_center = 0.0
        jitter = 0.25 if i % 2 == 0 else -0.25
        cones.append([x, y_center + track_width/2 + jitter]); blue_mask.append(True)
        cones.append([x, y_center - track_width/2 - jitter]); blue_mask.append(False)

    # Hairpin top (turning upward in y)
    ys = np.linspace(0.5, 6.5, 6)
    x_center = 10.5
    for i, y in enumerate(ys):
        jitter = 0.22 * np.cos(i)
        cones.append([x_center + track_width/2 + jitter, y]); blue_mask.append(True)
        cones.append([x_center - track_width/2 - jitter, y]); blue_mask.append(False)

    # Leg 2 (back in x)
    xs2 = np.linspace(10, 0, 7)
    y_center = 7.0
    for i, x in enumerate(xs2):
        jitter = 0.25 if i % 2 == 0 else -0.25
        cones.append([x, y_center + track_width/2 + jitter]); blue_mask.append(True)
        cones.append([x, y_center - track_width/2 - jitter]); blue_mask.append(False)

    cones = np.array(cones, dtype=float)
    colors = _colors_for(cones, np.array(blue_mask, dtype=bool))
    return cones, colors


def make_track_chicane(track_width=3.0):
    """
    Chicane / offset slalom: centerline shifts left then right.
    """
    xs = np.linspace(0, 24, 13)
    cones = []
    blue_mask = []

    for i, x in enumerate(xs):
        # piecewise center shift
        if x < 8:
            center_y = 0.0
        elif x < 16:
            center_y = 2.0
        else:
            center_y = -1.5

        jitter = 0.28 * np.sin(i)

        cones.append([x, center_y + track_width/2 + jitter]); blue_mask.append(True)
        cones.append([x, center_y - track_width/2 - jitter]); blue_mask.append(False)

    cones = np.array(cones, dtype=float)
    colors = _colors_for(cones, np.array(blue_mask, dtype=bool))
    return cones, colors



# Demo runner + plotting helpers

def _plot_top_paths(cones, colors, vehicle_pos, vehicle_heading, paths, costs, out_path, title):
    """
    Plots cones + top paths. Saves PNG to out_path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    labels = np.argmax(colors, axis=1)
    blue_mask = labels == cfg.CONE_COLOR_BLUE
    yellow_mask = labels == cfg.CONE_COLOR_YELLOW

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(cones[blue_mask, 0], cones[blue_mask, 1], s=50, label="Blue")
    ax.scatter(cones[yellow_mask, 0], cones[yellow_mask, 1], s=50, label="Yellow", edgecolors="black")

    ax.scatter(vehicle_pos[0], vehicle_pos[1], s=120, marker="s", label="Vehicle")
    ax.arrow(vehicle_pos[0], vehicle_pos[1],
             1.2*np.cos(vehicle_heading), 1.2*np.sin(vehicle_heading),
             head_width=0.3)

    # top 5 colors (fixed list so it’s consistent)
    path_colors = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown"]

    for i, path in enumerate(paths):
        p = np.asarray(path, dtype=float)
        lw = 3.5 if i == 0 else 1.8
        ax.plot(p[:, 0], p[:, 1], linewidth=lw, color=path_colors[i % len(path_colors)],
                label=f"#{i+1} cost={costs[i]:.3f}")

        # label start
        ax.text(p[0, 0], p[0, 1], f"{i+1}", fontsize=10)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _run_demo_track(track_name, cones, colors, vehicle_pos, vehicle_heading, max_depth=None, k_start=None, top_k=5):
    """
    Returns:
      candidate_paths, top_paths, top_costs
    """
    max_depth = cfg.MAX_TREE_DEPTH if max_depth is None else max_depth
    k_start = cfg.K_START if k_start is None else k_start

    candidate_paths = get_path_tree(cones, colors, vehicle_pos, vehicle_heading, max_depth, k_start)

    # If no candidates, return early (test will fail with a helpful message)
    if len(candidate_paths) == 0:
        return candidate_paths, [], []

    top_paths = beam_search_prune(candidate_paths, cones, colors, beam_width=top_k)
    top_costs = [evaluate_path_cost(np.array(p), cones, colors) for p in top_paths]

    # Sort to be safe (beam_search_prune should already do this)
    order = np.argsort(top_costs)
    top_paths = [top_paths[i] for i in order]
    top_costs = [top_costs[i] for i in order]

    # Print summary for demo logs
    print(f"\n--- {track_name} ---")
    print(f"candidates generated: {len(candidate_paths)}")
    print(f"top {len(top_paths)} kept:")
    for i, c in enumerate(top_costs):
        print(f"  #{i+1}: cost={c:.6f}  waypoints={len(top_paths[i])}")

    return candidate_paths, top_paths, top_costs


# Pytest tests

@pytest.mark.parametrize("name,track_fn", [
    ("straight_zigzag", make_track_straight_zigzag),
    ("gentle_s", make_track_gentle_s),
    ("hairpin", make_track_hairpin),
    ("chicane", make_track_chicane),
])
def test_demo_tracks_generate_and_plot(name, track_fn):
    cones, colors = track_fn()

    # Vehicle starts slightly behind first cones, facing +x
    vehicle_pos = np.array([-2.0, 0.0], dtype=float)
    vehicle_heading = 0.0

    candidates, top_paths, top_costs = _run_demo_track(
        name, cones, colors, vehicle_pos, vehicle_heading,
        top_k=5
    )

    # Assertions: demo should actually produce something
    assert len(candidates) > 0, f"{name}: no candidate paths generated"
    assert len(top_paths) > 0, f"{name}: beam search returned no paths"
    assert len(top_paths) <= 5
    assert all(np.isfinite(top_costs)), f"{name}: non-finite costs found"

    # Save plot
    out_path = os.path.join(os.path.dirname(__file__), "demo_outputs", f"{name}.png")
    _plot_top_paths(
        cones, colors, vehicle_pos, vehicle_heading,
        top_paths, top_costs,
        out_path,
        title=f"Demo Track: {name} (top {len(top_paths)})"
    )
