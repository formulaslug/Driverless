import numpy as np
from controller import StanleyController  # 这里按你实际文件名改
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class ControlVisualizer:
    """
    简单 HUD：一条转向条 + 一条油门/刹车条
    """
    def __init__(self, max_steer_rad: float):
        self.max_steer = max_steer_rad

        # 初始化图像
        plt.ion()
        self.fig, (self.ax_steer, self.ax_long) = plt.subplots(
            2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [1, 1]}
        )
        self.fig.suptitle("Stanley Controller HUD")

        # ---- 转向条 (水平) ----
        self.ax_steer.set_title("Steering")
        self.ax_steer.set_xlim(-1.0, 1.0)
        self.ax_steer.set_ylim(0, 1)
        self.ax_steer.axvline(0.0, color='black', linestyle='--', linewidth=1)  # 中线
        self.ax_steer.set_yticks([])
        self.ax_steer.set_xticks([-1, -0.5, 0, 0.5, 1])
        self.ax_steer.set_xlabel("Left  (-1)      0      (+1)  Right")

        # 我们用一个 rectangle 来表示当前转向
        self.steer_rect = Rectangle(
            (0, 0), 0, 1, alpha=0.6
        )
        self.ax_steer.add_patch(self.steer_rect)

        # ---- 油门/刹车条 (垂直) ----
        self.ax_long.set_title("Throttle / Brake")
        self.ax_long.set_xlim(0, 1)
        self.ax_long.set_ylim(-1.0, 1.0)
        self.ax_long.axhline(0.0, color='black', linestyle='--', linewidth=1)  # 分界线
        self.ax_long.set_xticks([])
        self.ax_long.set_yticks([-1, -0.5, 0, 0.5, 1])
        self.ax_long.set_ylabel("Brake (-1)   0   (+1) Throttle")

        # 这里用两块 rect：上面油门，下面刹车
        self.throttle_rect = Rectangle(
            (0.25, 0), 0.5, 0, alpha=0.6   # 从 0 往上长
        )
        self.brake_rect = Rectangle(
            (0.25, 0), 0.5, 0, alpha=0.6   # 从 0 往下长
        )
        self.ax_long.add_patch(self.throttle_rect)
        self.ax_long.add_patch(self.brake_rect)

        plt.tight_layout()

    def update(self, delta_cmd: float, throttle_cmd: float, brake_cmd: float):
        """
        更新 HUD 显示

        Parameters
        ----------
        delta_cmd : 转向角 [rad], 已经被 Stanley 饱和在 [-max_steer, +max_steer]
        throttle_cmd : [0, 1]
        brake_cmd : [0, 1]
        """

        # ---- 1. 归一化转向：[-1, 1] ----
        steer_norm = np.clip(delta_cmd / (self.max_steer + 1e-6), -1.0, 1.0)

        # rectangle 是从左边界开始的，我们让它从 0 伸到 steer_norm
        if steer_norm >= 0:
            self.steer_rect.set_x(0.0)
            self.steer_rect.set_width(steer_norm)
        else:
            self.steer_rect.set_x(steer_norm)
            self.steer_rect.set_width(-steer_norm)

        # ---- 2. 归一化纵向：油门为 +, 刹车为 - ----
        #   long_norm ∈ [-1, 1]
        long_norm = np.clip(throttle_cmd - brake_cmd, -1.0, 1.0)

        # throttle_rect：从 0 向上
        if long_norm > 0:
            self.throttle_rect.set_y(0.0)
            self.throttle_rect.set_height(long_norm)
        else:
            self.throttle_rect.set_y(0.0)
            self.throttle_rect.set_height(0.0)

        # brake_rect：从 0 向下（负方向）
        if long_norm < 0:
            self.brake_rect.set_y(long_norm)
            self.brake_rect.set_height(-long_norm)
        else:
            self.brake_rect.set_y(0.0)
            self.brake_rect.set_height(0.0)

        # ---- 3. 刷新图像 ----
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main_demo():
    """
    简单 demo：随机给 controller 喂点数据 + HUD 更新
    实际用的时候，你把这里换成你真实仿真/跑车循环就行
    """
    import time

    # 初始化你的 StanleyController（参数自己填）
    controller = StanleyController(
        k_e=1.0,
        k_yaw=1.0,
        k_v=1.0,
        max_steer=np.deg2rad(25.0),  # 例子：最大 25°
        k_throttle=0.3,
        k_brake=0.5,
        speed_deadband=0.2,
    )

    vis = ControlVisualizer(max_steer_rad=controller.max_steer)

    # 假装有一个循环（例如仿真每 0.05 s）
    x, y, yaw, v = 0.0, 0.0, 0.0, 0.0
    t = 0.0
    path_xy = np.array([[i, 0.0] for i in range(100)], dtype=float)  # 一条直线

    while True:
        v_ref = 5.0  # m/s，目标速度
        # 计算控制量
        delta_cmd, throttle_cmd, brake_cmd = controller.compute_control(
            x=x,
            y=y,
            yaw=yaw,
            v=v,
            path_xy=path_xy,
            v_ref=v_ref
        )

        # 更新 HUD
        vis.update(delta_cmd, throttle_cmd, brake_cmd)

        # 这里随便更新一下车辆状态（你自己会有更真实的动力学）
        v += (throttle_cmd - brake_cmd) * 0.1
        x += v * np.cos(yaw) * 0.05
        y += v * np.sin(yaw) * 0.05

        t += 0.05
        time.sleep(0.05)


if __name__ == "__main__":
    main_demo()
