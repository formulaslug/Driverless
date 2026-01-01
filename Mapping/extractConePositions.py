import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from Integration.coneSegmentor import ConeSegmentor
from Integration.depthEstimator import DepthEstimator
from Integration.distanceEstimator import DistanceEstimator
from DepthEstimation.ground_plane_ransac import estimateGroundPlane, getCameraIntrinsics

class ConePositionExtractor:
    def __init__(self, cameraConfigPath=None, coneConfigPath=None, exclusionAreas=None, depthScale=10.0):
        self.coneSegmentor = ConeSegmentor()
        self.depthEstimator = DepthEstimator()
        self.distanceEstimator = DistanceEstimator(cameraConfigPath=cameraConfigPath, coneConfigPath=coneConfigPath)
        self.exclusionAreas = exclusionAreas if exclusionAreas is not None else []
        self.depthScale = depthScale

        self.cameraIntrinsics = {
            'fx': self.distanceEstimator.fx,
            'fy': self.distanceEstimator.fy,
            'cx': self.distanceEstimator.cx,
            'cy': self.distanceEstimator.cy,
            'fov': self.distanceEstimator.fov
        }

    def extract3DPosition(self, box, distance):
        if distance is None or distance <= 0:
            return None

        x1, y1, x2, y2 = box
        u = (x1 + x2) / 2
        v = y2

        fx = self.cameraIntrinsics['fx']
        fy = self.cameraIntrinsics['fy']
        cx = self.cameraIntrinsics['cx']
        cy = self.cameraIntrinsics['cy']

        z = distance
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return {'x': x, 'y': y, 'z': z}

    def extractConePositions(self, image):
        segmentationResults = self.coneSegmentor.segment(image)

        if segmentationResults['numDetections'] == 0:
            return []

        depthMap = self.depthEstimator.estimateDepth(image)

        planeParams, inlierRatio, inlierMask = estimateGroundPlane(
            depthMap,
            self.cameraIntrinsics,
            exclusionAreas=self.exclusionAreas,
            depthScale=self.depthScale
        )

        boxes = segmentationResults['boxes']
        classes = segmentationResults['classes']
        confidences = segmentationResults['confidences']

        coneDistances = self.distanceEstimator.estimateAllCones(boxes, classes, depthMap, planeParams)

        conePositions = []
        for i in range(len(boxes)):
            box = boxes[i]
            coneClass = classes[i]
            confidence = confidences[i]
            distance = coneDistances[i]['distance']

            position3D = self.extract3DPosition(box, distance)

            if position3D is not None:
                conePositions.append({
                    'position': position3D,
                    'class': coneClass,
                    'confidence': float(confidence),
                    'box': box.tolist(),
                    'distance': distance,
                    'distanceMethods': {
                        'bbox': coneDistances[i]['bboxDistance'],
                        'depthMap': coneDistances[i]['depthMapDistance'],
                        'groundPlane': coneDistances[i]['groundPlaneDistance']
                    }
                })

        return conePositions

def main():
    import argparse
    import cv2
    import json

    parser = argparse.ArgumentParser(description='Extract cone positions relative to vehicle')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to output JSON file')
    parser.add_argument('--camera-config', type=str, default=None, help='Path to camera config JSON')
    parser.add_argument('--cone-config', type=str, default=None, help='Path to cone config JSON')
    parser.add_argument('--exclusions', type=str, default=None, help='Path to exclusion areas JSON')
    parser.add_argument('--depth-scale', type=float, default=10.0, help='Depth scale factor')

    args = parser.parse_args()

    exclusionAreas = None
    if args.exclusions is not None:
        with open(args.exclusions, 'r') as f:
            exclusionAreas = json.load(f)

    extractor = ConePositionExtractor(
        cameraConfigPath=args.camera_config,
        coneConfigPath=args.cone_config,
        exclusionAreas=exclusionAreas,
        depthScale=args.depth_scale
    )

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    conePositions = extractor.extractConePositions(image)

    print(f"Detected {len(conePositions)} cones:")
    for i, cone in enumerate(conePositions):
        pos = cone['position']
        print(f"  Cone {i+1}: {cone['class']['name']}")
        print(f"    Position: x={pos['x']:.2f}m, y={pos['y']:.2f}m, z={pos['z']:.2f}m")
        print(f"    Confidence: {cone['confidence']:.2f}")
        print(f"    Distance: {cone['distance']:.2f}m")

    if args.output is not None:
        with open(args.output, 'w') as f:
            json.dump(conePositions, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()
