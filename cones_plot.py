import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


## loading the json file
with open("/Users/atvyas/Desktop/coding/FS/sim_int/cones.json", "r") as f:
    data = json.load(f)

cones = data["cones"]
positions = np.array([cone["position"] for cone in cones])
colors = [cone["color"] for cone in cones]


## Drawing the cones:
def draw_cone(ax, x, y, z, height=0.4, radius=0.5, color="orange"):
    # sqaure base to the cones
    half = radius
    square = [
        [x - half, y - half, z],
        [x + half, y - half, z],
        [x + half, y + half, z],
        [x - half, y + half, z]
    ]

    base = Poly3DCollection(
        [square],
        facecolors=color,
        alpha=0.6
    )
    ax.add_collection3d(base)

    # the actual cones
    theta = np.linspace(0, 2 * np.pi, 1)
    r = np.linspace(0, radius, 2)
    T, R = np.meshgrid(theta, r)

    X = x + R * np.cos(T)
    Y = y + R * np.sin(T)
    Z = z + height * (1 - R / radius)

    ax.plot_surface(X, Y, Z, color=color, alpha=0.7, linewidth=0)


    # Cone surface
    theta = np.linspace(0, 2*np.pi, 5)
    r = np.linspace(0, radius, 2)
    T, R = np.meshgrid(theta, r)

    X = x + R * np.cos(T)
    Y = y + R * np.sin(T)
    Z = z + height * (1 - R / radius)

    ax.plot_surface(X, Y, Z, color=color, alpha=0.7, linewidth=0)


## Plotting the cones in 3-D
fig = plt.figure(figsize=(14, 7))

# 3d
ax3d = fig.add_subplot(121, projection="3d")

for (x, y, z), color in zip(positions, colors):
    draw_cone(ax3d, x, y, z, height=0.5, radius=0.5, color=color)

ax3d.set_title("3D Cone View")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")

ax3d.set_xlim(-6, 6)
ax3d.set_ylim(-6, 6)
ax3d.set_zlim(0, 3)

# top down
ax2d = fig.add_subplot(122)

x = positions[:, 0]
y = positions[:, 1]

ax2d.scatter(x, y, c=colors, s=120, edgecolors="black")

# setting up the grid
ax2d.set_title("Top-Down View (X-Y)")
ax2d.set_xlabel("X")
ax2d.set_ylabel("Y")
ax2d.set_aspect("equal")
ax2d.set_xlim(-6, 6)
ax2d.set_ylim(-6, 6)
ax2d.grid(True)

# to display the graphs
plt.tight_layout()
plt.show()
