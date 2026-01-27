import os
import sys
import json
import glob
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'Integration'))
from depthEstimator import DepthEstimator
from coneSegmentor import ConeSegmentor
from distanceEstimator import DistanceEstimator
from visualizationUtils import createFourTileVisualization

sys.path.append(os.path.join(os.path.dirname(__file__), 'DepthEstimation'))
from ground_plane_ransac import estimateGroundPlane, getCameraIntrinsics

def perception(image, depthEstimator, coneSegmentor, distEstimator, cameraIntrinsics):
    depthMap = depthEstimator.estimateDepth(image)
    segmentationResults = coneSegmentor.segment(image)

    planeParams, inlierRatio, inlierMask = estimateGroundPlane(
        depthMap,
        cameraIntrinsics,
        exclusionAreas=None,
        subsampleStep=5,
        inlierThreshold=0.1,
        maxTrials=100,
        maxTiltAngle=45,
        depthScale=10.0
    )

    coneDistances = distEstimator.estimateAllCones(
        segmentationResults['boxes'],
        segmentationResults['classes'],
        depthMap,
        planeParams
    )

    return depthMap, segmentationResults, planeParams, inlierMask, coneDistances

def main():
    print("Initializing models...")
    depthEstimator = DepthEstimator()
    coneSegmentor = ConeSegmentor()
    distEstimator = DistanceEstimator()

    framePattern = os.path.join('SampleData', 'driverless-10fps', 'frame_*.jpg')
    frameFiles = sorted(glob.glob(framePattern))

    outputDir = 'output'
    os.makedirs(outputDir, exist_ok=True)

    print(f"Processing {len(frameFiles)} frames...")

    for frameFile in frameFiles:
        frameName = os.path.splitext(os.path.basename(frameFile))[0]
        print(f"Processing {frameName}...")

        image = cv2.imread(frameFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]
        cameraIntrinsics = getCameraIntrinsics(W, H, fov=90)

        depthMap, segmentationResults, planeParams, inlierMask, coneDistances = perception(
            image, depthEstimator, coneSegmentor, distEstimator, cameraIntrinsics
        )

        depthPath = os.path.join(outputDir, f'depth_{frameName}.npy')
        np.save(depthPath, depthMap)

        segResults = {
            'numDetections': segmentationResults['numDetections'],
            'boxes': segmentationResults['boxes'].tolist() if segmentationResults['boxes'] is not None else None,
            'classes': segmentationResults['classes'],
            'confidences': segmentationResults['confidences'].tolist() if segmentationResults['confidences'] is not None else None
        }

        segPath = os.path.join(outputDir, f'seg_{frameName}.json')
        with open(segPath, 'w') as f:
            json.dump(segResults, f, indent=2)

        distancesOutput = []
        for dist in coneDistances:
            distancesOutput.append({
                'distance': dist['distance'],
                'bboxDistance': dist['bboxDistance'],
                'depthMapDistance': dist['depthMapDistance'],
                'groundPlaneDistance': dist['groundPlaneDistance'],
                'class': dist['class']
            })

        distPath = os.path.join(outputDir, f'distances_{frameName}.json')
        with open(distPath, 'w') as f:
            json.dump(distancesOutput, f, indent=2)

        visualization = createFourTileVisualization(
            image, depthMap, segmentationResults, planeParams, inlierMask,
            coneDistances, cameraIntrinsics, depthScale=10.0
        )

        visPath = os.path.join(outputDir, f'vis_{frameName}.png')
        visBgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(visPath, visBgr)

    print("Processing complete!")

if __name__ == "__main__":
    main()
