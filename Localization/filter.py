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
        cosYaw = np.cos(-dyaw)
        sinYaw = np.sin(-dyaw)

        for cone in self.cones:
            xRot = cone.x * cosYaw - cone.y * sinYaw
            yRot = cone.x * sinYaw + cone.y * cosYaw

            cone.x = xRot - dx
            cone.y = yRot - dy

    def getDominantClass(self, colorConfidences):
        maxConf = max(colorConfidences)
        if maxConf < 0.5:
            return -1
        return colorConfidences.index(maxConf)

    def areColorsCompatible(self, colorConf1, colorConf2):
        class1 = self.getDominantClass(colorConf1)
        class2 = self.getDominantClass(colorConf2)

        if class1 == -1 or class2 == -1:
            return True

        if class1 == class2:
            return True

        if (class1 == 2 and class2 == 3) or (class1 == 3 and class2 == 2):
            return True

        return False

    def matchDetections(self, detections):
        matches = []
        unmatchedDetections = []

        usedCones = set()

        for detection in detections:
            x, y, blueConf, yellowConf, sOrangeConf, lOrangeConf = detection
            colorConfidences = [blueConf, yellowConf, sOrangeConf, lOrangeConf]

            bestCone = None
            bestDistance = float('inf')

            for i, cone in enumerate(self.cones):
                if i in usedCones:
                    continue

                if not self.areColorsCompatible(colorConfidences, cone.colorConfidences):
                    continue

                dx = x - cone.x
                dy = y - cone.y
                distance = np.sqrt(dx*dx + dy*dy)

                if distance < self.matchThreshold and distance < bestDistance:
                    bestDistance = distance
                    bestCone = i

            if bestCone is not None:
                matches.append((bestCone, detection))
                usedCones.add(bestCone)
            else:
                unmatchedDetections.append(detection)

        return matches, unmatchedDetections

    def mergeConeWithDetection(self, cone, detection):
        x, y, blueConf, yellowConf, sOrangeConf, lOrangeConf = detection

        cone.x = (cone.x + x) / 2
        cone.y = (cone.y + y) / 2
        cone.age = 0
        cone.numObservations += 1

    def addNewCones(self, detections):
        for detection in detections:
            x, y, blueConf, yellowConf, sOrangeConf, lOrangeConf = detection
            colorConfidences = [blueConf, yellowConf, sOrangeConf, lOrangeConf]
            newCone = Cone(x, y, colorConfidences, self.initialVariance)
            self.cones.append(newCone)

    def update(self, detections, dx, dy, dyaw):
        self.transformConeLocations(dx, dy, dyaw)

        for cone in self.cones:
            cone.age += 1

        matches, unmatchedDetections = self.matchDetections(detections)

        for coneIdx, detection in matches:
            self.mergeConeWithDetection(self.cones[coneIdx], detection)

        self.addNewCones(unmatchedDetections)

        self.cones = [cone for cone in self.cones if cone.age <= self.maxAge]

    def visualize(self, savePath=None, showPlot=True):
        fig, ax = plt.subplots(figsize=(10, 10))

        coneColors = {
            0: '#0000FF',
            1: '#FFD700',
            2: '#FF8C00',
            3: '#FF4500',
        }

        vehicleSize = 0.5
        vehicleTriangle = plt.Polygon([
            [vehicleSize, 0],
            [0, vehicleSize * 0.4],
            [0, -vehicleSize * 0.4]
        ], color='red', fill=True, zorder=10)
        ax.add_patch(vehicleTriangle)

        for cone in self.cones:
            dominantClass = self.getDominantClass(cone.colorConfidences)
            if dominantClass == -1:
                color = '#808080'
            else:
                color = coneColors[dominantClass]

            ax.scatter(cone.x, cone.y, c=color, s=100, zorder=5, edgecolors='black', linewidths=1)

            ellipseWidth = 2 * np.sqrt(cone.varX)
            ellipseHeight = 2 * np.sqrt(cone.varY)
            ellipse = Ellipse((cone.x, cone.y),
                            width=ellipseWidth,
                            height=ellipseHeight,
                            facecolor='none',
                            edgecolor=color,
                            alpha=0.5,
                            linewidth=1.5)
            ax.add_patch(ellipse)

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Forward Distance (m)', fontsize=12)
        ax.set_ylabel('Lateral Distance (m)', fontsize=12)
        ax.set_title('Cone Map (Vehicle Frame)', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

        if len(self.cones) > 0:
            xCoords = [c.x for c in self.cones]
            yCoords = [c.y for c in self.cones]
            margin = 2.0
            xMin = min(xCoords + [0]) - margin
            xMax = max(xCoords + [vehicleSize]) + margin
            yMin = min(yCoords + [-vehicleSize * 0.4]) - margin
            yMax = max(yCoords + [vehicleSize * 0.4]) + margin
            ax.set_xlim(xMin, xMax)
            ax.set_ylim(yMin, yMax)
        else:
            ax.set_xlim(-2, 20)
            ax.set_ylim(-10, 10)

        if savePath is not None:
            plt.savefig(savePath, dpi=150, bbox_inches='tight')

        if showPlot:
            plt.show()
        else:
            plt.close()

    def getConeMap(self):
        return [(cone.x, cone.y, *cone.colorConfidences) for cone in self.cones]
