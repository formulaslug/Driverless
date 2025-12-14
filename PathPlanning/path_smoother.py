import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import config

def smooth_path(waypoints, num_points=100):
    # u: parameter values of original waypoints
    #    that measure how far along the path each waypoint is
    #    given as a normalized value [0,1]
    # tck: tuple(knots, coefficients, degree of spline)
    tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=0, k=3)

    # get num_points evenly spaced points
    u_fine = np.linspace(0, 1, num_points)

    # get smoothed x,y coordinates and path
    x_smooth, y_smooth = splev(u_fine, tck)
    smooth_path = np.column_stack([x_smooth, y_smooth])

    # get curvature (curvature = 1/radius)
    dx_dt, dy_dt = splev(u_fine, tck, der=1)
    d2x_dt2, d2y_dt2 = splev(u_fine, tck, der=2)

    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5

    return smooth_path, curvature

def visualize(test_cases):
    """
    Visualize smooth paths with curvature analysis.

    Args:
        test_cases: List of (name, waypoints) tuples
    """
    n = len(test_cases)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5*n), squeeze=False)

    for i, (name, waypoints) in enumerate(test_cases):
        # Smooth the path
        smooth, curvature = smooth_path(waypoints, num_points=200)

        # Calculate turn radius (clipped for visualization)
        radius = np.zeros_like(curvature)
        for j, k in enumerate(curvature):
            radius[j] = 1.0 / abs(k) if abs(k) > 0.001 else 100.0
        radius = np.clip(radius, 0, 50)

        # Calculate path progress percentage
        path_progress = np.linspace(0, 100, len(curvature))

        # Plot 1: Path with curvature coloring
        ax1 = axes[i, 0]
        scatter = ax1.scatter(smooth[:, 0], smooth[:, 1],
                            c=np.abs(curvature), cmap='RdYlGn_r',
                            s=20, vmin=0, vmax=0.3)
        ax1.plot(waypoints[:, 0], waypoints[:, 1], 'ko-',
                markersize=8, linewidth=1, alpha=0.5, label='Input waypoints')
        ax1.set_title(f'{name} - Smooth Path (colored by curvature)', fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('|Curvature| (1/m)')

        # Plot 2: Curvature profile
        ax2 = axes[i, 1]
        ax2.plot(path_progress, curvature, 'b-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.axhline(y=0.222, color='r', linestyle='--', linewidth=1.5,
                   label='Max allowed (9m diameter)')
        ax2.axhline(y=-0.222, color='r', linestyle='--', linewidth=1.5)
        ax2.fill_between(path_progress, -0.222, 0.222, alpha=0.2, color='green')
        ax2.set_title('Curvature Profile', fontweight='bold')
        ax2.set_xlabel('Path Progress (%)')
        ax2.set_ylabel('Curvature (1/m)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Turn radius
        ax3 = axes[i, 2]
        ax3.plot(path_progress, radius, 'g-', linewidth=2)
        ax3.fill_between(path_progress, radius, 50, alpha=0.3, color='green')
        ax3.axhline(y=4.5, color='r', linestyle='--', linewidth=1.5,
                   label='Min allowed (4.5m)')
        ax3.fill_between(path_progress, 0, 4.5, alpha=0.2, color='red')
        ax3.set_title('Turn Radius', fontweight='bold')
        ax3.set_xlabel('Path Progress (%)')
        ax3.set_ylabel('Radius (m)')
        ax3.set_ylim([0, 50])
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Print statistics
        max_curv = np.max(np.abs(curvature))
        min_radius = 1.0 / max_curv if max_curv > 0 else float('inf')
        status = "✓ PASS" if max_curv <= 0.222 else "✗ FAIL"
        print(f"{name:20s} | Max curvature: {max_curv:.4f} | Min radius: {min_radius:.2f}m | {status}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test track configurations
    straight_line = np.array([
        [0, 0], [5, 0], [10, 0], [15, 0], [20, 0]
    ])

    simple_curve = np.array([
        [0, 0], [5, 0], [10, 2], [15, 5], [20, 5]
    ])

    hairpin = np.array([
        [0, 0], [10, 0], [15, 5], [15, 10],
        [10, 15], [5, 15], [0, 10], [0, 5]
    ])

    slalom = np.array([
        [0, 0], [5, 2], [10, 0], [15, -2],
        [20, 0], [25, 2], [30, 0]
    ])

    oval_track = np.array([
        [0, 0], [10, 0], [20, 0], [25, 5], [25, 10],
        [20, 15], [10, 15], [0, 15], [-5, 10], [-5, 5]
    ])

    s_curve = np.array([
        [0, 0], [5, 2], [10, 5], [15, 7],
        [20, 7], [25, 5], [30, 2], [35, 0]
    ])

    # Test cases
    test_cases = [
        ("Straight Line", straight_line),
        ("Simple Curve", simple_curve),
        ("Hairpin Turn", hairpin),
        ("Slalom", slalom),
        ("Oval Track", oval_track),
        ("S-Curve", s_curve)
    ]

    # Visualize
    visualize(test_cases)
