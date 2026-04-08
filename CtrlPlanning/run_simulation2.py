import numpy as np
import matplotlib.pyplot as plt
import time
from controller import StanleyController
from visualize_controller import ControlVisualizer
from matplotlib.patches import Rectangle


def generate_short_path():
    # short path: straight then gentle right curve then straight
    A = np.array([[i, 0.0] for i in range(15)], dtype=float)
    angles = np.linspace(0.0, -np.deg2rad(30.0), 30)  # right turn
    R = 10.0
    B = np.stack([15.0 + R * np.sin(angles), R * (1 - np.cos(angles))], axis=1)
    C = np.array([[15.0 + R * np.sin(angles[-1]) + i, R * (1 - np.cos(angles[-1]))] for i in range(15)], dtype=float)
    return np.vstack([A, B, C])



def generate_random_path(num_points: int = 2000,
                         max_turn_deg: float = 120.0,
                         turn_std_deg: float = 10.0,
                         step_mean: float = 1.0,
                         step_std: float = 0.2,
                         seed: int | None = None) -> np.ndarray:
    """
    Generate a long random path where the heading change between consecutive
    steps does not exceed `max_turn_deg` degrees (absolute).

    Returns an array of shape (N,2) with absolute coordinates.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    max_turn = np.deg2rad(max_turn_deg)
    # start at origin heading along +x
    headings = np.zeros(num_points)
    # draw per-step heading changes limited to [-max_turn, max_turn]
    # use smaller typical turns by default (clamp at max_turn)
    delta_heading = rng.normal(loc=0.0, scale=np.deg2rad(turn_std_deg), size=num_points)
    delta_heading = np.clip(delta_heading, -max_turn, max_turn)
    headings = np.cumsum(delta_heading)

    # step lengths
    steps = np.clip(rng.normal(loc=step_mean, scale=step_std, size=num_points), 0.1, 3.0)

    xs = np.cumsum(steps * np.cos(headings))
    ys = np.cumsum(steps * np.sin(headings))

    return np.stack([xs, ys], axis=1)


def generate_racetrack_path(num_steps: int = 500,
                            step_mean: float = 1.0,
                            n_segments: int | None = None,
                            straight_len_mu: float = 20.0,
                            straight_len_std: float = 5.0,
                            arc_angle_mu_deg: float = 60.0,
                            arc_angle_std_deg: float = 15.0,
                            arc_radius_mu: float = 15.0,
                            arc_radius_std: float = 5.0,
                            seed: int | None = None) -> np.ndarray:
    """
    Generate a racetrack-like path composed of straight segments and circular arcs.

    - `num_steps` approximates how many points the path should contain (used to size segments).
    - The generator alternates straights and arcs to create distinct turns that resemble a track.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if n_segments is None:
        n_segments = max(4, int(num_steps / 80))

    pts = []
    x, y, heading = 0.0, 0.0, 0.0

    for seg in range(n_segments):
        # straight segment
        straight_len = max(5.0, rng.normal(loc=straight_len_mu, scale=straight_len_std))
        n_st = max(1, int(straight_len / step_mean))
        xs = x + np.cumsum(np.full(n_st, step_mean) * np.cos(heading))
        ys = y + np.cumsum(np.full(n_st, step_mean) * np.sin(heading))
        seg_pts = np.stack([xs, ys], axis=1)
        pts.append(seg_pts)
        x, y = seg_pts[-1]

        # arc (turn)
        angle_deg = max(15.0, rng.normal(loc=arc_angle_mu_deg, scale=arc_angle_std_deg))
        # alternate left/right turns to look like a track (with some randomness)
        turn_dir = -1 if (seg % 2 == 0) else 1
        if rng.random() < 0.15:
            turn_dir *= -1
        angle = np.deg2rad(angle_deg) * turn_dir
        radius = max(4.0, rng.normal(loc=arc_radius_mu, scale=arc_radius_std))

        arc_length = abs(angle) * radius
        n_arc = max(2, int(arc_length / step_mean))
        # center of the arc
        # center is located at (x, y) plus a vector perpendicular to heading
        cx = x - radius * np.sin(heading) * turn_dir
        cy = y + radius * np.cos(heading) * turn_dir

        # compute arc points
        start_angle = np.arctan2(y - cy, x - cx)
        arc_thetas = np.linspace(start_angle, start_angle + angle, n_arc + 1)[1:]
        ax_pts = cx + radius * np.cos(arc_thetas)
        ay_pts = cy + radius * np.sin(arc_thetas)
        arc_points = np.stack([ax_pts, ay_pts], axis=1)
        pts.append(arc_points)

        # update state to end of arc
        x, y = arc_points[-1]
        heading = (heading + angle) % (2 * np.pi)

    all_pts = np.vstack(pts)
    # resample so spacing is approximately step_mean
    dxy = np.diff(all_pts, axis=0)
    dists = np.hypot(dxy[:, 0], dxy[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(dists)])
    total_len = cum[-1]
    n_target = max(2, int(total_len / step_mean))
    new_s = np.linspace(0.0, total_len, n_target)
    new_x = np.interp(new_s, cum, all_pts[:, 0])
    new_y = np.interp(new_s, cum, all_pts[:, 1])
    return np.stack([new_x, new_y], axis=1)


def extend_path(path: np.ndarray,
                n_add: int = 200,
                max_turn_deg: float = 120.0,
                turn_std_deg: float = 8.0,
                step_mean: float = 1.0,
                step_std: float = 0.2,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Append `n_add` points to `path` continuing from its last heading.

    Returns the extended path (new array).
    """
    if rng is None:
        rng = np.random.default_rng()

    if path.shape[0] < 2:
        last_heading = 0.0
        last_x, last_y = 0.0, 0.0
    else:
        last_x, last_y = path[-1]
        prev_x, prev_y = path[-2]
        last_heading = float(np.arctan2(last_y - prev_y, last_x - prev_x))

    max_turn = np.deg2rad(max_turn_deg)
    delta_heading = rng.normal(loc=0.0, scale=np.deg2rad(turn_std_deg), size=n_add)
    delta_heading = np.clip(delta_heading, -max_turn, max_turn)
    headings = last_heading + np.cumsum(delta_heading)
    steps = np.clip(rng.normal(loc=step_mean, scale=step_std, size=n_add), 0.1, 3.0)

    xs = last_x + np.cumsum(steps * np.cos(headings))
    ys = last_y + np.cumsum(steps * np.sin(headings))

    new_pts = np.stack([xs, ys], axis=1)
    return np.vstack([path, new_pts])


def to_relative(path_xy, x, y, yaw):
    dx = path_xy[:, 0] - x
    dy = path_xy[:, 1] - y
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    x_local = dx * cos_y - dy * sin_y
    y_local = dx * sin_y + dy * cos_y
    return np.stack([x_local, y_local], axis=1)


def simple_dynamics(x, y, yaw, v, delta, throttle, brake, dt=0.05):
    # very simple bicycle-like update for demonstration
    a = throttle * 2.0 - brake * 3.0
    v = max(0.0, v + a * dt)
    x += v * np.cos(yaw) * dt
    y += v * np.sin(yaw) * dt
    # use a short wheelbase approximation
    L = 2.5
    yaw += v / L * np.tan(delta) * dt
    return x, y, yaw, v


def run(short=True,
    hud_mode: str = 'inset',
    autoscale: bool = True,
    num_steps: int | None = None,
    seed: int | None = 42,
    debug: bool = False,
    path_mode: str = 'random',
    start_state: tuple[float, float, float, float] | None = None,
    path_turn_std_deg: float = 10.0):
    """
    Run the 2D Stanley simulation.

    Parameters added:
    - path_mode: 'random' (default) or 'straight'. If 'straight', a straight-line
      path starting at the global origin along +X is used.
    - start_state: optional tuple (x, y, yaw, v) to set the vehicle initial state.
      If omitted, defaults to (0,0,0,0).
    """

    # choose path: random or straight
    if path_mode == 'straight':
        # straight line starting at global origin along +X
        if num_steps is not None:
            # create slightly longer path than steps with a small safety buffer
            length = max(int(num_steps * 1.1) + 50, 50)
        else:
            length = 400 if not short else 150
        path = np.array([[i, 0.0] for i in range(length)], dtype=float)
    elif path_mode == 'race':
        # racetrack-like path composed of straights + circular arcs
        if num_steps is None:
            num_steps = 500
        path = generate_racetrack_path(num_steps=num_steps, step_mean=1.0, seed=seed)
    else:
        # pass through `seed` so runs can be deterministic when desired
        if num_steps is not None:
            # generate one waypoint per step plus buffer (handles variable step sizes)
            num_points = max(int(num_steps * 1.2) + 100, 100)
        else:
            num_points = 300 if short else 3000

        path = generate_random_path(num_points=num_points,
                                    max_turn_deg=120.0,
                                    turn_std_deg=path_turn_std_deg,
                                    seed=seed)

    controller = StanleyController(
        k_e=0.5,
        k_yaw=0.5,
        k_v=1.0,
        max_steer=np.deg2rad(45.0),
        k_throttle=0.4,
        k_brake=0.4,
        speed_deadband=0.2,
    )

    # initial vehicle state (absolute coordinates)
    if start_state is None:
        x, y, yaw, v = 0.0, 0.0, 0.0, 0.0
    else:
        x, y, yaw, v = start_state
    v_ref = 4.0

    # prepare plots: 2D path + HUD
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 5))
    path_line, = ax.plot(path[:, 0], path[:, 1], '-k', linewidth=1.0, label='path')
    traj_line, = ax.plot([], [], '-r', linewidth=1.5, label='trajectory')
    veh_point, = ax.plot([], [], 'bo', label='vehicle')
    heading_arrow = None
    # marker for the waypoint Stanley used for this step
    waypoint_marker, = ax.plot([], [], marker='o', ms=10, mec='lime', mfc='none', linestyle='')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Stanley Controller - 2D Simulation')
    ax.legend()
    ax.grid(True)

    # step counter text (top-left, axes fraction)
    step_text = ax.text(0.01, 0.98, 'Step: 0', transform=ax.transAxes, va='top')

    # HUD: either separate window (ControlVisualizer) or inset axes inside main figure
    hud = None
    steer_rect = None
    throttle_rect = None
    brake_rect = None
    hud_ax = None
    step_text_hud = None

    # if the user clicks the figure (e.g., to zoom/pan), suspend autoscaling
    user_zoomed = False
    def _on_click(event):
        nonlocal user_zoomed
        user_zoomed = True
    fig.canvas.mpl_connect('button_press_event', _on_click)

    if hud_mode == 'separate':
        # independent figure (may open separate window)
        hud = ControlVisualizer(max_steer_rad=controller.max_steer)
    else:
        # inset HUD inside main figure (top-right corner)
        # coordinates: [left, bottom, width, height] in figure fraction
        hud_ax = fig.add_axes([0.68, 0.68, 0.28, 0.28])
        hud_ax.set_title('HUD')
        hud_ax.axis('off')
        # use normalized axes coordinates for patches (0..1) to make updates reliable
        hud_ax.set_xlim(0.0, 1.0)
        hud_ax.set_ylim(0.0, 1.0)
        # steering horizontal bar at top area (non-overlapping)
        steer_rect = Rectangle((0.1, 0.75), 0.0, 0.15, color='cyan', transform=hud_ax.transAxes)
        hud_ax.add_patch(steer_rect)

        # throttle (left) and brake (right) vertical bars in middle area
        throttle_rect = Rectangle((0.08, 0.1), 0.2, 0.0, color='green', transform=hud_ax.transAxes)
        brake_rect = Rectangle((0.72, 0.1), 0.2, 0.0, color='red', transform=hud_ax.transAxes)
        hud_ax.add_patch(throttle_rect)
        hud_ax.add_patch(brake_rect)
        # step counter inside HUD
        step_text_hud = hud_ax.text(0.02, 0.92, 'Step: 0', transform=hud_ax.transAxes, va='top')

    traj_x = []
    traj_y = []

    # vehicle state for visualization: current steering angle (rad)
    steer_angle = 0.0
    # simulation timestep (keep in sync with simple_dynamics dt)
    sim_dt = 0.05

    # markers to show the two points used to compute path heading
    heading_pt1_marker, = ax.plot([], [], marker='x', ms=8, mec='orange', mfc='none', linestyle='')
    heading_pt2_marker, = ax.plot([], [], marker='x', ms=8, mec='magenta', mfc='none', linestyle='')

    step = 0
    # main loop: run for `num_steps` iterations, or forever if num_steps is None
    while (num_steps is None) or (step < num_steps):
        # controller expects path in vehicle coordinates (vehicle at origin, yaw=0)
        local_path = to_relative(path, x, y, yaw)

        delta, throttle, brake = controller.compute_control(
            x=0.0, y=0.0, yaw=0.0, v=v,
            path_xy=local_path,
            v_ref=v_ref
        )

        # apply commanded steering as a delta to the current steering angle
        # Interpretation: controller returns a steering *change* (delta),
        # so we update the vehicle's current steer_angle by adding it.
        # Then saturate to controller limits so the vehicle can't exceed max_steer.
        steer_angle += float(delta)
        steer_angle = float(np.clip(steer_angle, -controller.max_steer, controller.max_steer))

        if debug:
            print(f"step={step} delta={delta:.4f} steer_angle={steer_angle:.4f} throttle={throttle:.3f} brake={brake:.3f} v={v:.3f}")

        # update HUD bars so user can see steering/throttle/brake
        if hud_mode == 'separate' and hud is not None:
            try:
                # pass vehicle's current steering angle (not raw controller delta)
                hud.update(steer_angle, throttle, brake)
            except Exception:
                pass
        elif hud_mode == 'inset' and hud_ax is not None:
            # invert steering sign so bar matches visualization direction
            steer_norm = np.clip(-steer_angle / (controller.max_steer + 1e-6), -1.0, 1.0)

            # steer bar: center region, draw to the right (positive) or left (negative)
            center_x = 0.5
            half_span = 0.4
            if steer_norm >= 0:
                # set xy and width in axes fraction
                steer_rect.set_xy((center_x, 0.75))
                steer_rect.set_width(half_span * steer_norm)
                steer_rect.set_height(0.15)
            else:
                steer_rect.set_xy((center_x + half_span * steer_norm, 0.75))
                steer_rect.set_width(-half_span * steer_norm)
                steer_rect.set_height(0.15)

            # throttle_brake: throttle grows upward from base y=0.1, brake grows upward from base y=0.1 on right side
            long_norm = np.clip(throttle - brake, -1.0, 1.0)
            base_y = 0.1
            max_h = 0.6
            # throttle at left patch
            if long_norm >= 0:
                throttle_rect.set_xy((0.08, base_y))
                throttle_rect.set_width(0.2)
                throttle_rect.set_height(max_h * long_norm)
                # clear brake
                brake_rect.set_xy((0.72, base_y))
                brake_rect.set_width(0.2)
                brake_rect.set_height(0.0)
            else:
                throttle_rect.set_xy((0.08, base_y))
                throttle_rect.set_width(0.2)
                throttle_rect.set_height(0.0)
                # brake shows proportionally on the right patch
                h = max_h * (-long_norm)
                brake_rect.set_xy((0.72, base_y))
                brake_rect.set_width(0.2)
                brake_rect.set_height(h)

        # use the controller's raw steering command for the vehicle update
        x, y, yaw, v = simple_dynamics(x, y, yaw, v, steer_angle, throttle, brake, dt=sim_dt)

        traj_x.append(x)
        traj_y.append(y)

        # extend path dynamically when vehicle approaches the end
        nearest_idx_abs = int(np.argmin((path[:, 0] - x)**2 + (path[:, 1] - y)**2))
        if nearest_idx_abs >= len(path) - 40:
            # append more points
            path = extend_path(path, n_add=800, max_turn_deg=120.0, turn_std_deg=path_turn_std_deg, step_mean=1.0, step_std=0.2)
            # update the drawn path line
            path_line.set_data(path[:, 0], path[:, 1])
        # determine which waypoint indices are used to compute the path heading
        # matches logic in `StanleyController.compute_control`: heading uses
        # (nearest_idx, nearest_idx+1) except at end where it uses (nearest_idx-1, nearest_idx)
        if nearest_idx_abs < len(path) - 1:
            i1, i2 = nearest_idx_abs, nearest_idx_abs + 1
        else:
            i1, i2 = max(0, nearest_idx_abs - 1), nearest_idx_abs

        # highlight the nominal nearest waypoint (used for cross-track)
        try:
            wp_x, wp_y = float(path[nearest_idx_abs, 0]), float(path[nearest_idx_abs, 1])
            waypoint_marker.set_data([wp_x], [wp_y])
        except Exception:
            waypoint_marker.set_data([], [])

        # highlight the two points used to compute path heading
        try:
            p1x, p1y = float(path[i1, 0]), float(path[i1, 1])
            p2x, p2y = float(path[i2, 0]), float(path[i2, 1])
            heading_pt1_marker.set_data([p1x], [p1y])
            heading_pt2_marker.set_data([p2x], [p2y])
        except Exception:
            heading_pt1_marker.set_data([], [])
            heading_pt2_marker.set_data([], [])

        # update plots every 2 steps to keep interactive responsive
        if step % 2 == 0:
            traj_line.set_data(traj_x, traj_y)
            veh_point.set_data([x], [y])
            # update step counter text
            try:
                step_text.set_text(f"Step: {step}")
            except Exception:
                pass
            if step_text_hud is not None:
                try:
                    step_text_hud.set_text(f"Step: {step}")
                except Exception:
                    pass

            # draw heading arrow (remove previous)
            if heading_arrow is not None:
                try:
                    heading_arrow.remove()
                except Exception:
                    pass
            dx = 0.8 * np.cos(yaw)
            dy = 0.8 * np.sin(yaw)
            heading_arrow = ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc='b', ec='b')

            # autoscale view to include trajectory and path (unless user zoomed or autoscale disabled)
            if autoscale and not user_zoomed:
                all_x = np.concatenate([path[:, 0], np.array(traj_x)])
                all_y = np.concatenate([path[:, 1], np.array(traj_y)])
                xmin, xmax = np.min(all_x) - 2.0, np.max(all_x) + 2.0
                ymin, ymax = np.min(all_y) - 2.0, np.max(all_y) + 2.0
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            fig.canvas.draw()
            fig.canvas.flush_events()

            # update HUD draw as well (in case backend needs flushing)
            if hud_mode == 'separate' and hud is not None:
                try:
                    hud.fig.canvas.draw()
                    hud.fig.canvas.flush_events()
                except Exception:
                    pass
            elif hud_mode == 'inset' and hud_ax is not None:
                try:
                    hud_ax.figure.canvas.draw()
                    hud_ax.figure.canvas.flush_events()
                except Exception:
                    pass

        time.sleep(0.02)
        step += 1

    # keep the final plot open
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    # run indefinitely by default when executed directly
    run(short=True, hud_mode='inset', autoscale=True, num_steps=1000, seed=43, debug=True, path_mode='race', path_turn_std_deg=15)
