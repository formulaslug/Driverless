from controller import Controller
from longitudinal.gain_scheduler import Gains
from longitudinal.longitudinal import Longitudinal
import numpy as np
import matplotlib.pyplot as plt


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Sims-Data")))
from FullVehichleSim.paramLoader import *
from FullVehichleSim.engine import dynamicStepState


steps = 1000
gains = Gains(0.0, 0.0, 0.0, 0.0)
controller = Controller(K_p=gains.KP, K_i=gains.KI, K_d=gains.KD)

arr1 = np.array(
    [0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0]
)

data = np.zeros((1000, arr1.shape[0]))
data[0, :] = arr1

for step in range(1, data.shape[0]):
    data[step] = dynamicStepState(data[step-1])

t = np.arange(data.shape[0]) / 100