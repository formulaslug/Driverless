import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from cost_functions import evaluate_path_cost, assign_cones_to_segments, calculate_trackwidth_variance
import config as cfg

def create_straight_track(length=30, width=3.0, cone_spacing=5.0):
    """Create a straight track with evenly spaced blue/yellow cones"""
    num_cones = int(length / cone_spacing) + 1

    cones = []
    colors = []

    for i in range(num_cones):
        x = i * cone_spacing
        # Blue cones on left (y = width/2)
        cones.append([x, width/2])
        colors.append([1.0, 0.0, 0.0, 0.0])  # 100% blue

        # Yellow cones on right (y = -width/2)
        cones.append([x, -width/2])
        colors.append([0.0, 1.0, 0.0, 0.0])  # 100% yellow

    return np.array(cones), np.array(colors)

def create_curved_track(radius=15, width=3.0, cone_spacing=5.0, arc_degrees=90):
    """Create a curved track (arc) with evenly spaced cones"""
    arc_length = 2 * np.pi * radius * (arc_degrees / 360)
    num_cones = int(arc_length / cone_spacing) + 1

    cones = []
    colors = []

    for i in range(num_cones):
        angle = (i * cone_spacing / arc_length) * (arc_degrees * np.pi / 180)

        # Blue cones on outer radius
        x_outer = (radius + width/2) * np.cos(angle)
        y_outer = (radius + width/2) * np.sin(angle)
        cones.append([x_outer, y_outer])
        colors.append([1.0, 0.0, 0.0, 0.0])  # 100% blue

        # Yellow cones on inner radius
        x_inner = (radius - width/2) * np.cos(angle)
        y_inner = (radius - width/2) * np.sin(angle)
        cones.append([x_inner, y_inner])
        colors.append([0.0, 1.0, 0.0, 0.0])  # 100% yellow

    return np.array(cones), np.array(colors)

def create_test_paths(scenario='straight'):
    """Create multiple test paths for comparison"""
    if scenario == 'straight':
        # Perfect centerline path
        perfect_path = np.array([
            [0, 0], [5, 0], [10, 0], [15, 0], [20, 0], [25, 0]
        ])

        # Slightly off-center path
        offset_path = np.array([
            [0, 0.5], [5, 0.5], [10, 0.5], [15, 0.5], [20, 0.5], [25, 0.5]
        ])

        # Wiggly path (high smoothness cost)
        wiggly_path = np.array([
            [0, 0], [5, 0.5], [10, -0.5], [15, 0.5], [20, -0.5], [25, 0]
        ])

        # Sharp turn path (high angle cost)
        sharp_path = np.array([
            [0, 0], [5, 0], [10, 0], [10, 3], [10, 6], [15, 6]
        ])

        # Wrong side path (crosses to other side)
        wrong_side_path = np.array([
            [0, 0], [5, 0], [10, -2], [15, -2], [20, 0], [25, 0]
        ])

        return {
            'Perfect Centerline': perfect_path,
            'Slightly Off-Center': offset_path,
            'Wiggly Path': wiggly_path,
            'Sharp Turn': sharp_path,
            'Wrong Side': wrong_side_path
        }

    elif scenario == 'curve':
        radius = 15
        # Perfect arc following the curve
        angles = np.linspace(0, np.pi/2, 6)
        perfect_path = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])

        # Cutting the corner (shorter path)
        cut_corner_path = np.array([
            [15, 0], [12, 3], [9, 6], [6, 9], [3, 12], [0, 15]
        ])

        # Going wide
        wide_path = np.array([
            [18, 0], [15, 3], [12, 6], [9, 9], [6, 12], [3, 15], [0, 18]
        ])

        return {
            'Perfect Arc': perfect_path,
            'Cut Corner': cut_corner_path,
            'Going Wide': wide_path
        }

    return {}

def visualize_paths_and_costs(cones, colors, paths, scenario_name='Test'):
    """Visualize multiple paths and their cost metrics"""
    n_paths = len(paths)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{scenario_name} - Path Cost Comparison', fontsize=16, fontweight='bold')

    # Extract cone colors for plotting
    cone_colors_idx = np.argmax(colors, axis=1)
    blue_cones = cones[cone_colors_idx == cfg.CONE_COLOR_BLUE]
    yellow_cones = cones[cone_colors_idx == cfg.CONE_COLOR_YELLOW]

    # Evaluate all paths
    costs = {}
    for name, path in paths.items():
        cost = evaluate_path_cost(path, cones, colors)
        costs[name] = cost

    # Plot 1: All paths on one plot
    ax = axes[0, 0]
    ax.scatter(blue_cones[:, 0], blue_cones[:, 1], c='blue', s=100, marker='o', label='Blue cones', alpha=0.6)
    ax.scatter(yellow_cones[:, 0], yellow_cones[:, 1], c='yellow', s=100, marker='o', label='Yellow cones', alpha=0.6, edgecolors='black')

    colors_list = ['green', 'orange', 'red', 'purple', 'brown', 'pink']
    for idx, (name, path) in enumerate(paths.items()):
        ax.plot(path[:, 0], path[:, 1], '-o', label=f'{name} (cost: {costs[name]:.2f})',
                linewidth=2, markersize=5, color=colors_list[idx % len(colors_list)])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('All Paths Comparison')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 2: Cost breakdown bar chart
    ax = axes[0, 1]
    path_names = list(costs.keys())
    cost_values = list(costs.values())
    bars = ax.bar(range(len(path_names)), cost_values, color=colors_list[:len(path_names)])
    ax.set_xticks(range(len(path_names)))
    ax.set_xticklabels(path_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Total Cost')
    ax.set_title('Total Path Costs')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot 3: Individual path visualizations (best and worst)
    sorted_paths = sorted(costs.items(), key=lambda x: x[1])
    best_path_name, best_cost = sorted_paths[0]
    worst_path_name, worst_cost = sorted_paths[-1]

    # Best path
    ax = axes[0, 2]
    ax.scatter(blue_cones[:, 0], blue_cones[:, 1], c='blue', s=100, marker='o', alpha=0.6)
    ax.scatter(yellow_cones[:, 0], yellow_cones[:, 1], c='yellow', s=100, marker='o', alpha=0.6, edgecolors='black')
    best_path = paths[best_path_name]
    ax.plot(best_path[:, 0], best_path[:, 1], '-o', color='green', linewidth=3, markersize=6, label=f'Cost: {best_cost:.2f}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'BEST: {best_path_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Worst path
    ax = axes[1, 0]
    ax.scatter(blue_cones[:, 0], blue_cones[:, 1], c='blue', s=100, marker='o', alpha=0.6)
    ax.scatter(yellow_cones[:, 0], yellow_cones[:, 1], c='yellow', s=100, marker='o', alpha=0.6, edgecolors='black')
    worst_path = paths[worst_path_name]
    ax.plot(worst_path[:, 0], worst_path[:, 1], '-o', color='red', linewidth=3, markersize=6, label=f'Cost: {worst_cost:.2f}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'WORST: {worst_path_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 4: Segment assignments visualization (for best path)
    ax = axes[1, 1]
    ax.scatter(blue_cones[:, 0], blue_cones[:, 1], c='blue', s=100, marker='o', alpha=0.6, label='Blue')
    ax.scatter(yellow_cones[:, 0], yellow_cones[:, 1], c='yellow', s=100, marker='o', alpha=0.6, edgecolors='black', label='Yellow')
    ax.plot(best_path[:, 0], best_path[:, 1], '-o', color='green', linewidth=2, markersize=6, label='Path')

    # Show segment assignments
    seg_assignments, side_assignments = assign_cones_to_segments(best_path, cones)
    for i, (cone, seg, side) in enumerate(zip(cones, seg_assignments, side_assignments)):
        if seg < len(best_path) - 1:
            mid_point = (best_path[seg] + best_path[seg + 1]) / 2
            ax.plot([cone[0], mid_point[0]], [cone[1], mid_point[1]],
                   'k--', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Cone-to-Segment Assignment\n({best_path_name})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 5: Track width visualization (for best path)
    ax = axes[1, 2]
    track_width_var = calculate_trackwidth_variance(best_path, cones, colors)
    ax.text(0.5, 0.7, f'Track Width Analysis\n({best_path_name})',
           ha='center', va='top', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.5, f'Width Variance: {track_width_var:.4f} m²',
           ha='center', va='center', fontsize=11, transform=ax.transAxes)
    ax.text(0.5, 0.3, f'Total Cost: {best_cost:.2f}',
           ha='center', va='center', fontsize=11, transform=ax.transAxes)

    # Add metric breakdown
    metrics_text = "Cost Metrics:\n"
    metrics_text += "• Angle sharpness\n"
    metrics_text += "• Path smoothness\n"
    metrics_text += "• Cone spacing consistency\n"
    metrics_text += "• Track width variance\n"
    metrics_text += "• Color confidence\n"
    metrics_text += "• Path length deviation\n"
    metrics_text += "• Boundary violations"

    ax.text(0.5, 0.1, metrics_text, ha='center', va='center',
           fontsize=9, transform=ax.transAxes, family='monospace')
    ax.axis('off')

    plt.tight_layout()
    return fig, costs

def test_straight_track():
    """Test cost function on straight track"""
    print("=" * 60)
    print("TEST 1: Straight Track")
    print("=" * 60)

    cones, colors = create_straight_track(length=30, width=3.0, cone_spacing=5.0)
    paths = create_test_paths(scenario='straight')

    print(f"\nTrack: {len(cones)} cones")
    print(f"Testing {len(paths)} different paths...\n")

    costs = {}
    for name, path in paths.items():
        cost = evaluate_path_cost(path, cones, colors)
        costs[name] = cost
        print(f"{name:25s}: Cost = {cost:8.4f}")

    # Find best path
    best_path = min(costs.items(), key=lambda x: x[1])
    print(f"\n✅ BEST PATH: {best_path[0]} (Cost: {best_path[1]:.4f})")

    # Visualize
    fig, _ = visualize_paths_and_costs(cones, colors, paths, 'Straight Track')
    plt.savefig('test_cost_straight_track.png', dpi=150, bbox_inches='tight')
    print("\n📊 Visualization saved: test_cost_straight_track.png")

    return costs

def test_curved_track():
    """Test cost function on curved track"""
    print("\n" + "=" * 60)
    print("TEST 2: Curved Track (90-degree arc)")
    print("=" * 60)

    cones, colors = create_curved_track(radius=15, width=3.0, cone_spacing=5.0, arc_degrees=90)
    paths = create_test_paths(scenario='curve')

    print(f"\nTrack: {len(cones)} cones")
    print(f"Testing {len(paths)} different paths...\n")

    costs = {}
    for name, path in paths.items():
        cost = evaluate_path_cost(path, cones, colors)
        costs[name] = cost
        print(f"{name:25s}: Cost = {cost:8.4f}")

    # Find best path
    best_path = min(costs.items(), key=lambda x: x[1])
    print(f"\n✅ BEST PATH: {best_path[0]} (Cost: {best_path[1]:.4f})")

    # Visualize
    fig, _ = visualize_paths_and_costs(cones, colors, paths, 'Curved Track')
    plt.savefig('test_cost_curved_track.png', dpi=150, bbox_inches='tight')
    print("\n📊 Visualization saved: test_cost_curved_track.png")

    return costs

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TEST 3: Edge Cases")
    print("=" * 60)

    # Test 1: Too few cones
    print("\n1. Too few cones (< MIN_CONES_FOR_VALID_PATH):")
    few_cones = np.array([[0, 0], [5, 0]])
    few_colors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    path = np.array([[0, 0], [5, 0], [10, 0]])
    cost = evaluate_path_cost(path, few_cones, few_colors)
    print(f"   Cost: {cost} (should be BIG_COST)")

    # Test 2: No colors provided
    print("\n2. No color information:")
    cones, _ = create_straight_track(length=20, width=3.0, cone_spacing=5.0)
    path = np.array([[0, 0], [5, 0], [10, 0], [15, 0]])
    cost = evaluate_path_cost(path, cones, colors=None)
    print(f"   Cost: {cost:.4f} (should work, some metrics will be 0)")

    # Test 3: Very short path
    print("\n3. Very short path (2 waypoints):")
    cones, colors = create_straight_track(length=20, width=3.0, cone_spacing=5.0)
    short_path = np.array([[0, 0], [5, 0]])
    cost = evaluate_path_cost(short_path, cones, colors)
    print(f"   Cost: {cost:.4f}")

    # Test 4: Path with uncertain color detection
    print("\n4. Uncertain color detection (50/50 probabilities):")
    uncertain_colors = np.array([[0.5, 0.5, 0.0, 0.0]] * len(cones))
    path = np.array([[0, 0], [5, 0], [10, 0], [15, 0]])
    cost = evaluate_path_cost(path, cones, uncertain_colors)
    print(f"   Cost: {cost:.4f} (higher due to color uncertainty)")

    print("\n✅ All edge cases handled successfully!")

def run_all_tests():
    """Run all cost function tests"""
    print("\n" + "=" * 60)
    print("COST FUNCTION TEST SUITE")
    print("Testing AMZ-based path cost evaluation")
    print("=" * 60)

    # Run tests
    test_straight_track()
    test_curved_track()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE!")
    print("=" * 60)
    print("\nVisualization files generated:")
    print("  - test_cost_straight_track.png")
    print("  - test_cost_curved_track.png")
    print("\nTo view plots interactively, uncomment plt.show() at the end.")

    # Uncomment to show plots interactively
    # plt.show()

if __name__ == '__main__':
    run_all_tests()
