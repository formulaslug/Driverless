import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import linear_sum_assignment

class Cone:
    def __init__(self, x, y, colorConfidences, initialVariance=0.25):
        self.x = x
        self.y = y
        self.cov = np.array([[initialVariance, 0.0],
                             [0.0, initialVariance]])
        self.colorConfidences = colorConfidences
        self.age = 0
        self.numObservations = 1

class ConeFilter:
    def __init__(self,
                 xMovementUncertainty=0.05,
                 yMovementUncertainty=0.05,
                 deltaYawUncertainty=0.02,
                 initialVariance=0.25,
                 measurementVariance=0.25,
                 matchThreshold=1.0,
                 mahalanobisThreshold=3.0,
                 colorMatchThreshold=0.2,
                 maxAge=10,
                 useMahalanobis=True):
        self.xMovementUncertainty = xMovementUncertainty
        self.yMovementUncertainty = yMovementUncertainty
        self.deltaYawUncertainty = deltaYawUncertainty
        self.initialVariance = initialVariance
        self.measurementVariance = measurementVariance
        self.matchThreshold = matchThreshold
        self.mahalanobisThreshold = mahalanobisThreshold
        self.colorMatchThreshold = colorMatchThreshold
        self.maxAge = maxAge
        self.useMahalanobis = useMahalanobis

        self.cones = []

    def transformConeLocations(self, dx, dy, dyaw):
        # p_new = R(-dyaw) @ (p_old - [dx, dy])
        cosYaw = np.cos(-dyaw)
        sinYaw = np.sin(-dyaw)

        J = np.array([[cosYaw, -sinYaw],
                      [sinYaw, cosYaw]])

        dxRot = dx * cosYaw - dy * sinYaw
        dyRot = dx * sinYaw + dy * cosYaw

        for cone in self.cones:
            xRot = cone.x * cosYaw - cone.y * sinYaw
            yRot = cone.x * sinYaw + cone.y * cosYaw

            cone.x = xRot - dxRot
            cone.y = yRot - dyRot

            covRotated = J @ cone.cov @ J.T

            # Yaw uncertainty applied only in tangential direction
            dist = np.sqrt(cone.x**2 + cone.y**2)
            if dist > 1e-6:
                angle = np.arctan2(cone.y, cone.x)
                tangentX = -np.sin(angle)
                tangentY = np.cos(angle)
                tangent = np.array([[tangentX], [tangentY]])
                Qyaw = (dist * self.deltaYawUncertainty)**2 * (tangent @ tangent.T)
            else:
                Qyaw = np.zeros((2, 2))

            Q = np.array([[self.xMovementUncertainty**2, 0.0],
                         [0.0, self.yMovementUncertainty**2]]) + Qyaw

            cone.cov = covRotated + Q

    def normalizeColorConfidences(self, colorConfidences):
        total = sum(colorConfidences)
        if total > 0:
            return [c / total for c in colorConfidences]
        else:
            return [1.0 / len(colorConfidences)] * len(colorConfidences)

    def colorConfidenceSimilarity(self, colorConf1, colorConf2):
        dotProduct = sum(c1 * c2 for c1, c2 in zip(colorConf1, colorConf2))
        return dotProduct

    def getDominantClass(self, colorConfidences):
        maxConf = max(colorConfidences)
        if maxConf < 0.5:
            return -1
        return colorConfidences.index(maxConf)

    def areColorsCompatible(self, colorConf1, colorConf2):
        similarity = self.colorConfidenceSimilarity(colorConf1, colorConf2)

        if similarity >= self.colorMatchThreshold:
            return True

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
        if len(detections) == 0 or len(self.cones) == 0:
            return [], list(detections)

        R = np.eye(2) * self.measurementVariance

        # Build cost matrix: rows=detections, cols=cones
        numDetections = len(detections)
        numCones = len(self.cones)
        sentinel = 1e9
        costMatrix = np.full((numDetections, numCones), sentinel)

        for di, detection in enumerate(detections):
            x, y, blueConf, yellowConf, sOrangeConf, lOrangeConf = detection
            colorConfidences = [blueConf, yellowConf, sOrangeConf, lOrangeConf]

            for ci, cone in enumerate(self.cones):
                if not self.areColorsCompatible(colorConfidences, cone.colorConfidences):
                    continue

                delta = np.array([x - cone.x, y - cone.y])

                if self.useMahalanobis:
                    S = cone.cov + R
                    Sinv = np.linalg.inv(S)
                    distance = np.sqrt(delta @ Sinv @ delta)
                    threshold = self.mahalanobisThreshold
                else:
                    distance = np.sqrt(delta @ delta)
                    threshold = self.matchThreshold

                if distance < threshold:
                    costMatrix[di, ci] = distance

        rowInds, colInds = linear_sum_assignment(costMatrix)

        matches = []
        matchedDetectionInds = set()
        for ri, ci in zip(rowInds, colInds):
            if costMatrix[ri, ci] < sentinel:
                matches.append((ci, detections[ri]))
                matchedDetectionInds.add(ri)

        unmatchedDetections = [detections[i] for i in range(numDetections) if i not in matchedDetectionInds]

        return matches, unmatchedDetections

    def mergeConeWithDetection(self, cone, detection):
        x, y, blueConf, yellowConf, sOrangeConf, lOrangeConf = detection
        newColorConfidences = self.normalizeColorConfidences([blueConf, yellowConf, sOrangeConf, lOrangeConf])

        R = np.eye(2) * self.measurementVariance
        S = cone.cov + R
        K = cone.cov @ np.linalg.inv(S)

        innovation = np.array([x - cone.x, y - cone.y])
        update = K @ innovation
        cone.x += update[0]
        cone.y += update[1]

        cone.cov = (np.eye(2) - K) @ cone.cov

        alpha = 1.0 / (cone.numObservations + 1)
        fusedConfidences = [
            (1 - alpha) * cone.colorConfidences[i] + alpha * newColorConfidences[i]
            for i in range(len(cone.colorConfidences))
        ]
        cone.colorConfidences = self.normalizeColorConfidences(fusedConfidences)

        cone.age = 0
        cone.numObservations += 1

    def addNewCones(self, detections):
        for detection in detections:
            x, y, blueConf, yellowConf, sOrangeConf, lOrangeConf = detection
            colorConfidences = self.normalizeColorConfidences([blueConf, yellowConf, sOrangeConf, lOrangeConf])
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

            # Extract ellipse angle from full covariance
            eigvals, eigvecs = np.linalg.eigh(cone.cov)
            angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
            ellipseWidth = 2 * np.sqrt(eigvals[1])
            ellipseHeight = 2 * np.sqrt(eigvals[0])
            ellipse = Ellipse((cone.x, cone.y),
                            width=ellipseWidth,
                            height=ellipseHeight,
                            angle=angle,
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
