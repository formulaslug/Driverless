import json
import numpy as np
import os

class DistanceEstimator:
    def __init__(self, cameraConfigPath=None, coneConfigPath=None, cameraIntrinsics=None):
        if coneConfigPath is None:
            coneConfigPath = os.path.join(os.path.dirname(__file__), 'cone.json')

        self.loadConeConfig(coneConfigPath)

        if cameraIntrinsics is not None:
            self.fx = cameraIntrinsics['fx']
            self.fy = cameraIntrinsics['fy']
            self.cx = cameraIntrinsics['cx']
            self.cy = cameraIntrinsics['cy']
            self.fov = cameraIntrinsics.get('fov', 120)
            self.depthScale = cameraIntrinsics.get('depthScale', 10.0)
        else:
            if cameraConfigPath is None:
                cameraConfigPath = os.path.join(os.path.dirname(__file__), 'camera.json')
            self.loadCameraConfig(cameraConfigPath)

    def loadCameraConfig(self, configPath):
        try:
            with open(configPath, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Camera config not found: {configPath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in camera config {configPath}: {e}")

        self.width = config['resolution']['width']
        self.height = config['resolution']['height']
        self.fov = config['fov']
        self.depthScale = config.get('depthScale', 10.0)

        if self.fov <= 0 or self.fov >= 180:
            raise ValueError(f"Invalid FOV {self.fov}: must be between 0 and 180 degrees")

        if config['fx'] is not None and config['fy'] is not None:
            self.fx = config['fx']
            self.fy = config['fy']
        else:
            focalLength = self.width / (2 * np.tan(np.radians(self.fov) / 2))
            self.fx = focalLength
            self.fy = focalLength

        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Invalid focal lengths fx={self.fx}, fy={self.fy}: must be positive")

        if config['cx'] is not None and config['cy'] is not None:
            self.cx = config['cx']
            self.cy = config['cy']
        else:
            self.cx = self.width / 2
            self.cy = self.height / 2

    def loadConeConfig(self, configPath):
        try:
            with open(configPath, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cone config not found: {configPath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in cone config {configPath}: {e}")

        self.coneHeights = config['coneHeights']

    def getConeHeight(self, coneClass):
        if isinstance(coneClass, dict):
            className = coneClass.get('name', 'seg_unknown_cone')
        else:
            className = coneClass

        return self.coneHeights.get(className, 0.228)

    def estimateDistanceBbox(self, box, coneClass):
        if len(box) != 4:
            return None

        x1, y1, x2, y2 = box
        bboxHeight = y2 - y1

        if bboxHeight <= 0:
            return None

        coneHeight = self.getConeHeight(coneClass)
        distance = (coneHeight * self.fy) / bboxHeight

        return distance if distance > 0 else None

    def sampleDisparityForBox(self, box, depthMap):
        if len(box) != 4:
            return None

        try:
            x1, y1, x2, y2 = map(int, box)
        except (ValueError, TypeError):
            return None

        H, W = depthMap.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return None

        marginX = int(width * 0.33)
        marginY = int(height * 0.33)

        innerX1 = max(x1, min(x1 + marginX, x2))
        innerX2 = max(x1, min(x2 - marginX, x2))
        innerY1 = max(y1, min(y1 + marginY, y2))
        innerY2 = max(y1, min(y2 - marginY, y2))

        if innerX2 <= innerX1 or innerY2 <= innerY1:
            return None

        depthRegion = depthMap[innerY1:innerY2, innerX1:innerX2]
        validDepths = depthRegion[(depthRegion > 0) & np.isfinite(depthRegion)]

        if len(validDepths) == 0:
            return None

        meanDisparity = float(np.mean(validDepths))
        return meanDisparity if meanDisparity != 0 else None

    def estimateDistanceDepthMap(self, box, depthMap, depthScale=None):
        if depthScale is None:
            depthScale = self.depthScale

        meanDisparity = self.sampleDisparityForBox(box, depthMap)
        if meanDisparity is None:
            return None

        return depthScale / meanDisparity

    def calibrateDepthScale(self, boxes, classes, depthMap):
        if boxes is None or len(boxes) == 0:
            return self.depthScale

        candidateScales = []

        for i in range(len(boxes)):
            bboxDist = self.estimateDistanceBbox(boxes[i], classes[i])
            if bboxDist is None or bboxDist <= 0:
                continue

            meanDisparity = self.sampleDisparityForBox(boxes[i], depthMap)
            if meanDisparity is None:
                continue

            candidateScales.append(bboxDist * meanDisparity)

        if len(candidateScales) < 2:
            return self.depthScale

        return float(np.median(candidateScales))

    def estimateDistanceGroundPlane(self, box, planeParams):
        if planeParams is None or len(planeParams) != 4:
            return None

        if self.fx == 0 or self.fy == 0:
            return None

        x1, y1, x2, y2 = box
        a, b, c, d = planeParams

        u = (x1 + x2) / 2
        v = y2

        xNorm = (u - self.cx) / self.fx
        yNorm = (v - self.cy) / self.fy

        denominator = a * xNorm + b * yNorm + c

        if abs(denominator) < 1e-6:
            return None

        distance = d / denominator

        return distance if distance > 0 else None

    def estimateDistance(self, box, coneClass, depthMap, planeParams, depthScale=None):
        weights = [1/3, 1/3, 1/3]

        distBbox = self.estimateDistanceBbox(box, coneClass)
        distDepth = self.estimateDistanceDepthMap(box, depthMap, depthScale=depthScale)
        distPlane = self.estimateDistanceGroundPlane(box, planeParams)

        distances = [distBbox, distDepth, distPlane]
        validDistances = []
        validWeights = []

        for d, w in zip(distances, weights):
            if d is not None and d > 0 and np.isfinite(d):
                validDistances.append(d)
                validWeights.append(w)

        if not validDistances:
            return None, distances

        totalWeight = sum(validWeights)
        if totalWeight == 0:
            return None, distances

        finalDistance = sum(d * w for d, w in zip(validDistances, validWeights)) / totalWeight

        return finalDistance, distances

    def estimateAllCones(self, boxes, classes, depthMap, planeParams, depthScale=None):
        if boxes is None or len(boxes) == 0:
            return []

        coneDistances = []

        for i in range(len(boxes)):
            box = boxes[i]
            coneClass = classes[i]

            distance, methodDistances = self.estimateDistance(box, coneClass, depthMap, planeParams, depthScale=depthScale)

            coneDistances.append({
                'distance': distance,
                'bboxDistance': methodDistances[0],
                'depthMapDistance': methodDistances[1],
                'groundPlaneDistance': methodDistances[2],
                'class': coneClass
            })

        return coneDistances
