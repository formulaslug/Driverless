import numpy as np
import matplotlib.pyplot as plt
from state_test import (
    position, speed, heading,
    acceleration, throttle, brakes, lastCurrent
)

# sim time set up
dt = 0.1           # time step (seconds)
t_end = 10.0       # total simulation time
time = np.arange(0, t_end, dt)

# sim values over sim time
speed_t = speed + acceleration * time
heading_t = heading + 0.02 * time          # slow turn
throttle_t = np.full_like(time, throttle)
brakes_t = np.full_like(time, brakes)
current_t = lastCurrent + 0.5 * time       # example trend

# Position integration
x = position[0] + np.cumsum(speed_t * np.cos(heading_t) * dt)
y = position[1] + np.cumsum(speed_t * np.sin(heading_t) * dt)

## plots
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
axs = axs.flatten()

axs[0].plot(time, speed_t)
axs[0].set_title("Speed vs Time")
axs[0].set_ylabel("m/s")

axs[1].plot(time, heading_t)
axs[1].set_title("Heading vs Time")
axs[1].set_ylabel("radians")

axs[2].plot(time, throttle_t)
axs[2].set_title("Throttle vs Time")

axs[3].plot(time, brakes_t)
axs[3].set_title("Brakes vs Time")

axs[4].plot(time, current_t)
axs[4].set_title("Current vs Time")
axs[4].set_ylabel("Amps")

axs[5].plot(x, y)
axs[5].set_title("Position (X vs Y)")
axs[5].set_xlabel("X")
axs[5].set_ylabel("Y")

for ax in axs:
    ax.set_xlabel("Time (s)")
    ax.grid(True)

plt.tight_layout()
plt.show()
