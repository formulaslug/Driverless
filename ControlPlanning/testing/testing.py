import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from controller import Controller

def make_straight_curve(length=20.0, spacing=0.5):
    n = int(length / spacing)
    xs = np.linspace(0, length, n)
    ys = np.zeros(n)
    return list(zip(xs, ys))


def make_s_curve(length=60.0, amplitude=5.0, wavelength=30.0, spacing=0.5):
    n = int(length / spacing)
    xs = np.linspace(0, length, n)
    ys = amplitude * np.sin(2 * np.pi * xs / wavelength)
    return list(zip(xs, ys))

if __name__ == "__main__":
    test = Controller(K_p=0.5, K_i=0.1, K_d=0.01)
    waypoints = make_s_curve()

    x = y = 1 
    yaw = 45
    v = 2
    t = 0.01

    for i in range(100):
        a_des, curv = test.update(x=x, y=y, yaw=45, v=v, t=0.01, v_desired=5, waypoints=waypoints)
        x += 1
        y += 1
        v += 0.1
        t += 1

        print(f'a_des : {a_des} || k : {curv}')