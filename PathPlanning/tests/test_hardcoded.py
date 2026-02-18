import numpy as np
import pytest
import config as cfg
from path_tree import get_path_tree
from beam_search import beam_search_prune
from cost_functions import evaluate_path_cost

import matplotlib.pyplot as plt


# Colors helper
def _colors_for(cones, blue_mask):
    colors = np.zeros((len(cones), 4), dtype=float)
    colors[blue_mask, cfg.CONE_COLOR_BLUE] = 0.95
    colors[~blue_mask, cfg.CONE_COLOR_YELLOW] = 0.95
    colors[:, 2] = 0.02
    colors[:, 3] = 0.01
    colors = colors / np.sum(colors, axis=1, keepdims=True)
    return colors


def _make_corridor(xs, center_y_fn, width=3.2, noise_std=0.00):
    """
    xs: array of x positions
    center_y_fn: function mapping x -> center_y

    Returns cones, colors.
    """
    cones = []
    blue_mask = []

    for x in xs:
        c = float(center_y_fn(x))

        cones.append([x, c + width / 2])
        blue_mask.append(True)

        cones.append([x, c - width / 2])
        blue_mask.append(False)

    cones = np.array(cones, dtype=float)
    colors = _colors_for(cones, np.array(blue_mask))
    return cones, colors


# 4 Demo Tracks (simple but "Delaunay-friendly")
def make_track_straight():
    xs = np.linspace(0, 28, 16)
    return _make_corridor(xs, center_y_fn=lambda x: 0.0)


def make_track_gentle_s():
    xs = np.linspace(0, 28, 18)
    return _make_corridor(xs, center_y_fn=lambda x: 1.0 * np.sin(x / 5.0))


def make_track_chicane():
    xs = np.linspace(0, 30, 20)

    def center(x):
        if x < 10:
            return 0.0
        elif x < 18:
            return 1.6
        else:
            return -1.0

    return _make_corridor(xs, center_y_fn=center)


def make_track_hairpin_like():
    xs = np.linspace(0, 32, 22)

    def center(x):
        # ramp upward after ~16m to simulate a hard turn
        if x < 16:
            return 0.0
        t = (x - 16) / (32 - 16)
        return 6.0 * (t**1.2)

    return _make_corridor(xs, center_y_fn=center)


# Plotting
def _plot_top_paths(cones, colors, vehicle_pos, vehicle_heading, paths, costs, title):
    labels = np.argmax(colors, axis=1)
    blue_mask = labels == cfg.CONE_COLOR_BLUE
    yellow_mask = labels == cfg.CONE_COLOR_YELLOW

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(cones[blue_mask, 0], cones[blue_mask, 1], s=50, label="Blue")
    ax.scatter(cones[yellow_mask, 0], cones[yellow_mask, 1], s=50, label="Yellow", edgecolors="black")

    ax.scatter(vehicle_pos[0], vehicle_pos[1], s=140, marker="s", label="Vehicle")
    ax.arrow(vehicle_pos[0], vehicle_pos[1],
             1.4*np.cos(vehicle_heading), 1.4*np.sin(vehicle_heading),
             head_width=0.3)

    path_colors = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown"]

    for i, path in enumerate(paths):
        p = np.asarray(path, dtype=float)
        lw = 4.0 if i == 0 else 2.2
        ax.plot(p[:, 0], p[:, 1], linewidth=lw, color=path_colors[i % len(path_colors)],
                label=f"#{i+1} cost={costs[i]:.3f}")

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    plt.show()

# Demo runner
def _run_demo_track(track_name, cones, colors, vehicle_pos, vehicle_heading, max_depth=20, k_start=3, top_k=5):
    coordinate_confidence = np.zeros(len(cones))
    candidate_paths = get_path_tree(cones, coordinate_confidence, colors, vehicle_pos, vehicle_heading, max_depth, k_start)

    if len(candidate_paths) == 0:
        print(f"\n--- {track_name} ---")
        print("NO candidates generated.")
        return candidate_paths, [], []

    top_paths = beam_search_prune(candidate_paths, cones, coordinate_confidence, colors, beam_width=top_k)
    top_costs = [evaluate_path_cost(np.array(p), cones, coordinate_confidence, colors) for p in top_paths]

    order = np.argsort(top_costs)
    top_paths = [top_paths[i] for i in order]
    top_costs = [top_costs[i] for i in order]

    print(f"\n--- {track_name} ---")
    print(f"candidates generated: {len(candidate_paths)}")
    print(f"top {len(top_paths)} kept:")
    for i, c in enumerate(top_costs):
        print(f"  #{i+1}: cost={c:.6f}  waypoints={len(top_paths[i])}")

    return candidate_paths, top_paths, top_costs


# Pytest
@pytest.mark.parametrize("name,track_fn", [
    ("straight", make_track_straight),
    ("gentle_s", make_track_gentle_s),
    ("chicane", make_track_chicane),
    ("hairpin_like", make_track_hairpin_like),
])
def test_demo_tracks_generate_and_plot(name, track_fn):
    cones, colors = track_fn()
    coordinate_confidence = np.zeros(len(cones))

    vehicle_pos = np.array([-1.0, 0.0], dtype=float)
    vehicle_heading = 0.0

    candidates, top_paths, top_costs = _run_demo_track(
        name, cones, colors, vehicle_pos, vehicle_heading,
        max_depth=30, k_start=3, top_k=5
    )

    assert len(candidates) > 0, f"{name}: no candidate paths generated"
    assert len(top_paths) > 0, f"{name}: beam search returned no paths"
    assert len(top_paths) <= 5
    assert all(np.isfinite(top_costs)), f"{name}: non-finite costs found"

    _plot_top_paths(
        cones, colors, vehicle_pos, vehicle_heading,
        top_paths, top_costs,
        title=f"Demo Track: {name} (top {len(top_paths)})"
    )
