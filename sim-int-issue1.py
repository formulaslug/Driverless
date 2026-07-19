
'''Take vechile postion, speed, and last current as input from state.py and use numpy to find the
 postions of the cones and the car and graph it on a coordinate system.'''

import numpy as np
import matplotlib.pyplot as plt

# Veichel state form state.py
stepSize = 0.1
position = np.array([12.5, 7.8])   # (x, y)
speed = 18.2                       # m/s
heading = np.deg2rad(15)           # radians
acceleration = 1.3                 # m/s^2
throttle = 0.65                    # 0–1
brakes = 0.1                       # 0–1
lastCurrent = 42.0                 # amps

# cones postions coming from a json file
cones = np.array([
    [5, 5],
    [8, 6],
    [10, 7],
    [13, 8],
    [16, 9],
    [18, 10]
])

# finding velcoity from heading
velocity = np.array([
    np.cos(heading),
    np.sin(heading)
]) * speed

#creating layout
fig, axs = plt.subplots(3, 1, figsize=(9, 12))

# cones and psition
axs[0].scatter(cones[:, 0], cones[:, 1],
               marker="^", s=150, label="Cones")

axs[0].scatter(position[0], position[1],
               marker="o", s=200, label="Vehicle")

axs[0].arrow(
    position[0], position[1],
    velocity[0] * 0.2, velocity[1] * 0.2,
    head_width=0.4,
    length_includes_head=True,
    label="Velocity"
)

axs[0].text(
    position[0] + 0.5,
    position[1] + 0.5,
    f"Speed: {speed:.1f} m/s",
    fontsize=10
)

axs[0].set_title("Vehicle & Cone Map")
axs[0].set_xlabel("X Position (m)")
axs[0].set_ylabel("Y Position (m)")
axs[0].axis("equal")
axs[0].grid(True)
axs[0].legend()

#Motion data
axs[1].bar(
    ["Speed (m/s)", "Acceleration (m/s²)"],
    [speed, acceleration]
)

axs[1].set_title("Vehicle Motion State")
axs[1].grid(True)

#electrical data
axs[2].bar(
    ["Throttle", "Brakes", "Motor Current (A)"],
    [throttle, brakes, lastCurrent]
)

axs[2].set_title("Control & Electrical State")
axs[2].grid(True)

plt.tight_layout()
plt.show()
