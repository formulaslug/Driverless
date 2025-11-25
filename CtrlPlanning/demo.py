import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from controller import StanleyController   # 你的文件

# ============================================================
# 1. HUD 可视化
# ============================================================
class ControlVisualizer:
    def __init__(self, max_steer_rad):
        self.max_steer = max_steer_rad

        plt.ion()
        self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4))
        self.ax_steer = ax1
        self.ax_long = ax2

        # ---- Steering bar ----
        self.ax_steer.set_title("Steering")
        self.ax_steer.set_xlim(-1, 1)
        self.ax_steer.set_ylim(0, 1)
        self.ax_steer.axvline(0, color='black')
        self.ax_steer.set_yticks([])
        self.steer_rect = Rectangle((0, 0), 0, 1, color='cyan')
        self.ax_steer.add_patch(self.steer_rect)

        # ---- Longitudinal ----
        self.ax_long.set_title("Throttle / Brake")
        self.ax_long.set_xlim(0, 1)
        self.ax_long.set_ylim(-1, 1)
        self.ax_long.axhline(0, color='black')
        self.ax_long.set_xticks([])

        self.throttle_rect = Rectangle((0.25, 0), 0.5, 0, color='green')
        self.brake_rect = Rectangle((0.25, 0), 0.5, 0, color='red')
        self.ax_long.add_patch(self.throttle_rect)
        self.ax_long.add_patch(self.brake_rect)

    def update(self, delta, throttle, brake):
        steer_norm = np.clip(delta / self.max_steer, -1, 1)

        # steering
        if steer_norm >= 0:
            self.steer_rect.set_x(0)
            self.steer_rect.set_width(steer_norm)
        else:
            self.steer_rect.set_x(steer_norm)
            self.steer_rect.set_width(-steer_norm)

        # throttle / brake
        long_norm = throttle - brake
        self.throttle_rect.set_height(max(long_norm, 0))
        self.brake_rect.set_y(min(long_norm, 0))
        self.brake_rect.set_height(abs(min(long_norm, 0)))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ============================================================
# 2. 构造 C 型赛道：直线 → 左弯 → 直线
# ============================================================
def generate_track_C():
    """
    Absolute coordinate path.
    然后会在每一帧转成 relative coordinate。
    """

    # segment A: straight
    A = np.array([[i, 0] for i in range(20)])

    # segment B: left turn, radius 15m, 60 degrees arc
    R = 15
    angles = np.linspace(0, np.deg2rad(60), 40)
    B = np.stack([
        20 + R * np.sin(angles),
        0  + R * (1 - np.cos(angles))
    ], axis=1)

    # segment C: straight after turn
    C = np.array([[20 + R*np.sin(np.deg2rad(60)) + i,
                   R*(1 - np.cos(np.deg2rad(60)))] for i in range(20)])

    return np.vstack([A, B, C])

# ============================================================
# 3. 把 absolute waypoints 转到车坐标
# ============================================================
def to_relative(path_xy, x, y, yaw):
    """
    车辆永远是 (0,0, yaw=0)，waypoints 转到 local coordinate.
    """
    dx = path_xy[:, 0] - x
    dy = path_xy[:, 1] - y
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)

    # rotation
    x_local = dx * cos_y - dy * sin_y
    y_local = dx * sin_y + dy * cos_y
    return np.stack([x_local, y_local], axis=1)

# ============================================================
# 4. 简单动力学（非常简化）
# ============================================================
def simple_dynamics(x, y, yaw, v, delta, throttle, brake, dt=0.05):
    a = throttle*2.0 - brake*3.0
    v = max(0, v + a*dt)
    x += v * np.cos(yaw) * dt
    y += v * np.sin(yaw) * dt
    yaw += v/2.5 * np.tan(delta) * dt
    return x, y, yaw, v

# ============================================================
# 5. 主模拟程序
# ============================================================
def main():
    track = generate_track_C()

    controller = StanleyController(
        k_e=1.0,
        k_yaw=1.2,
        k_v=1.0,
        max_steer=np.deg2rad(25),
        k_throttle=0.4,
        k_brake=0.4,
        speed_deadband=0.3
    )

    vis = ControlVisualizer(max_steer_rad=controller.max_steer)

    x, y, yaw, v = 0.0, 0.0, 0.0, 0.0
    v_ref = 4.0

    for _ in range(2000):
        # 转到 relative coordinate（你的 controller 需要这个）
        local_path = to_relative(track, x, y, yaw)

        delta, throttle, brake = controller.compute_control(
            x=0, y=0, yaw=0, v=v,
            path_xy=local_path,
            v_ref=v_ref
        )

        vis.update(delta, throttle, brake)

        # 更新 absolute vehicle state
        x, y, yaw, v = simple_dynamics(x, y, yaw, v, delta, throttle, brake)

        time.sleep(0.05)


if __name__ == "__main__":
    main()
