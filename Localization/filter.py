import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Cone:
    def __init__(self, x, y, colorConfidences, initialVariance=0.25):
        self.x = x
        self.y = y
        self.varX = initialVariance
        self.varY = initialVariance
        self.colorConfidences = colorConfidences
        self.age = 0
        self.numObservations = 1

class ConeFilter:
    def __init__(self,
                 xMovementUncertainty=0.05,
                 yMovementUncertainty=0.05,
                 deltaYawUncertainty=0.02,
                 initialVariance=0.25,
                 matchThreshold=1.0,
                 mahalanobisThreshold=3.0,
                 colorMatchThreshold=0.2,
                 maxAge=10):
        self.xMovementUncertainty = xMovementUncertainty
        self.yMovementUncertainty = yMovementUncertainty
        self.deltaYawUncertainty = deltaYawUncertainty
        self.initialVariance = initialVariance
        self.matchThreshold = matchThreshold
        self.mahalanobisThreshold = mahalanobisThreshold
        self.colorMatchThreshold = colorMatchThreshold
        self.maxAge = maxAge

        self.cones = []

    def transformConeLocations(self, dx, dy, dyaw):
        cosYaw = np.cos(-d
                            yaw)
        sinYaw = np.sin(-dyaw)

        for cone in self.cones:
            xRot = cone.x * cosYaw - cone.y * sinYaw
            yRot = cone.x * sinYaw + cone.y * cosYaw

            cone.x = xRot - dx
            cone.y = yRot - dy

    def addNewCones(self, detections):
        for detection in detections:
            x, y, blueConf, yellowConf, sOrangeConf, lOrangeConf = detection
            colorConfidences = [blueConf, yellowConf, sOrangeConf, lOrangeConf]
            newCone = Cone(x, y, colorConfidences, self.initialVariance)
            self.cones.append(newCone)

    def update(self, detections, dx, dy, dyaw):
        self.transformConeLocations(dx, dy, dyaw)

        self.addNewCones(detections)

    def getConeMap(self):
        return [(cone.x, cone.y, *cone.colorConfidences) for cone in self.cones]
