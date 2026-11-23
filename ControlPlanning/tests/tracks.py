import numpy as np

# Simple test tracks, all starting at the origin heading +x (vehicle frame)


def straight_path(length=20.0, spacing=0.5, y=0.0):
    xs = np.arange(0.0, length + spacing, spacing)
    return np.column_stack([xs, np.full_like(xs, y)])


def s_curve(length=60.0, amplitude=2.5, wavelength=30.0, spacing=0.5):
    """Left-right weave that starts tangent to the car's heading, like a real planned path."""
    xs = np.arange(0.0, length + spacing, spacing)
    return np.column_stack([xs, amplitude * (1 - np.cos(2 * np.pi * xs / wavelength))])


def arc(radius, sweep_rad, left=True, spacing=0.5):
    """Constant-radius turn starting at the origin heading +x."""
    n = max(int(radius * sweep_rad / spacing), 2)
    theta = np.linspace(0.0, sweep_rad, n)
    sign = 1.0 if left else -1.0
    return np.column_stack([radius * np.sin(theta), sign * radius * (1 - np.cos(theta))])

