import json
import os
import numpy as np

pixelSigma = 2.0
pitchSigmaRad = np.radians(0.5)
borderMargin = 2
maxRange = 100.0
disagreementSigmas = 3.0
defaultConeHeight = 0.325

# Per-cone position from a pinhole camera: bbox height gives depth along the
# optical axis, and the base pixel's ray against the flat road gives a second
# estimate. The two are fused by inverse variance, falling back to the lower
# variance one when they disagree.
class DistanceEstimator:
    def __init__(self, camera, coneConfigPath=None):
        if coneConfigPath is None:
            coneConfigPath = os.path.join(os.path.dirname(__file__), 'cone.json')
        self.camera = camera
        self.loadConeConfig(coneConfigPath)

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
        className = coneClass.get('name', 'seg_unknown_cone') if isinstance(coneClass, dict) else coneClass
        return self.coneHeights.get(className, defaultConeHeight)

    # Returns ((x, y), sigmaX) or None. Needs the full cone height visible.
    def estimateBbox(self, box, coneClass):
        x1, y1, x2, y2 = box
        pixelHeight = y2 - y1
        if pixelHeight <= 0 or y1 <= borderMargin or y2 >= self.camera.height - 1 - borderMargin:
            return None

        camera = self.camera
        depth = self.getConeHeight(coneClass) * camera.fy * np.cos(camera.pitchRad) / pixelHeight
        u = (x1 + x2) / 2.0
        point = camera.pointAtDepth(u, y2, depth)
        forwardPerDepth = camera.rayDirection(u, y2)[0]
        sigmaX = abs(forwardPerDepth) * depth * pixelSigma / pixelHeight
        return point, sigmaX

    # Returns ((x, y), sigmaX) or None. Needs the cone base visible.
    def estimateGround(self, box):
        x1, y1, x2, y2 = box
        if y2 >= self.camera.height - 1 - borderMargin:
            return None

        camera = self.camera
        u = (x1 + x2) / 2.0
        point = camera.groundPoint(u, y2)
        if point is None or point[0] <= 0 or point[0] > maxRange:
            return None

        samples = [camera.groundPoint(u, y2 + pixelSigma),
                   camera.groundPoint(u, y2, camera.pitchRad + pitchSigmaRad)]
        deviations = [abs(s[0] - point[0]) if s is not None else maxRange for s in samples]
        return point, float(np.hypot(*deviations))

    def fuse(self, bbox, ground):
        if bbox is None and ground is None:
            return None
        if bbox is None or ground is None:
            return (bbox or ground)[0]

        (bboxPoint, bboxSigma), (groundPoint, groundSigma) = bbox, ground
        gap = abs(bboxPoint[0] - groundPoint[0])
        if gap > disagreementSigmas * np.hypot(bboxSigma, groundSigma):
            return bboxPoint if bboxSigma <= groundSigma else groundPoint

        bboxWeight = 1.0 / max(bboxSigma, 1e-6) ** 2
        groundWeight = 1.0 / max(groundSigma, 1e-6) ** 2
        total = bboxWeight + groundWeight
        return (
            (bboxWeight * bboxPoint[0] + groundWeight * groundPoint[0]) / total,
            (bboxWeight * bboxPoint[1] + groundWeight * groundPoint[1]) / total,
        )

    def estimateCone(self, box, coneClass):
        bbox = self.estimateBbox(box, coneClass)
        ground = self.estimateGround(box)
        fused = self.fuse(bbox, ground)

        valid = fused is not None and fused[0] > 0 and np.isfinite(fused[0]) and np.isfinite(fused[1])
        return {
            'x': float(fused[0]) if valid else None,
            'y': float(fused[1]) if valid else None,
            'distance': float(fused[0]) if valid else None,
            'bboxDistance': bbox[0][0] if bbox else None,
            'groundDistance': ground[0][0] if ground else None,
            'class': coneClass,
        }

    def estimateAllCones(self, boxes, classes):
        if boxes is None or len(boxes) == 0:
            return []
        return [self.estimateCone(boxes[i], classes[i]) for i in range(len(boxes))]
