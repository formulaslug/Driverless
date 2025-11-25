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
    delta_heading = rng.normal(loc=0.0, scale=np.deg2rad(10.0), size=num_points)
    delta_heading = np.clip(delta_heading, -max_turn, max_turn)
    headings = np.cumsum(delta_heading)

    # step lengths
    steps = np.clip(rng.normal(loc=step_mean, scale=step_std, size=num_points), 0.1, 3.0)

    xs = np.cumsum(steps * np.cos(headings))
    ys = np.cumsum(steps * np.sin(headings))

    return np.stack([xs, ys], axis=1)


def extend_path(path: np.ndarray,
                n_add: int = 200,
                max_turn_deg: float = 120.0,
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
    delta_heading = rng.normal(loc=0.0, scale=np.deg2rad(8.0), size=n_add)
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


def run(short=True, hud_mode: str = 'inset', autoscale: bool = True, num_steps: int | None = None):
    # choose path: generate a random path (short or long)
    if short:
        path = generate_random_path(num_points=300, max_turn_deg=120.0, seed=42)
    else:
        path = generate_random_path(num_points=3000, max_turn_deg=120.0, seed=42)

    controller = StanleyController(
        k_e=1.0,
        k_yaw=1.2,
        k_v=1.0,
        max_steer=np.deg2rad(25.0),
        k_throttle=0.4,
        k_brake=0.4,
        speed_deadband=0.2,
    )

    # initial vehicle state (absolute coordinates)
    x, y, yaw, v = 0.0, 0.0, 0.0, 0.0
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

        # update HUD bars so user can see steering/throttle/brake
        if hud_mode == 'separate' and hud is not None:
            try:
                hud.update(delta, throttle, brake)
            except Exception:
                pass
        elif hud_mode == 'inset' and hud_ax is not None:
            # invert steering sign so bar matches visualization direction
            steer_norm = np.clip(-delta / (controller.max_steer + 1e-6), -1.0, 1.0)

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

        x, y, yaw, v = simple_dynamics(x, y, yaw, v, delta, throttle, brake, dt=0.05)

        traj_x.append(x)
        traj_y.append(y)

        # extend path dynamically when vehicle approaches the end
        nearest_idx_abs = int(np.argmin((path[:, 0] - x)**2 + (path[:, 1] - y)**2))
        if nearest_idx_abs >= len(path) - 40:
            # append more points
            path = extend_path(path, n_add=800, max_turn_deg=120.0, step_mean=1.0, step_std=0.2)
            # update the drawn path line
            path_line.set_data(path[:, 0], path[:, 1])
        # determine which waypoint the Stanley controller actually used for nearest_point
        # The controller uses nearest_idx, except when nearest_idx == last index, it uses previous point
        if nearest_idx_abs < len(path) - 1:
            idx_to_highlight = nearest_idx_abs
        else:
            idx_to_highlight = max(0, nearest_idx_abs - 1)
        try:
            wp_x, wp_y = float(path[idx_to_highlight, 0]), float(path[idx_to_highlight, 1])
            waypoint_marker.set_data([wp_x], [wp_y])
        except Exception:
            waypoint_marker.set_data([], [])

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
    run(short=True, hud_mode='inset', autoscale=True, num_steps=None)
