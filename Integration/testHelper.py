import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from depthEstimator import DepthEstimator
from coneSegmentor import ConeSegmentor
from distanceEstimator import DistanceEstimator
from visualizationUtils import createFourTileVisualization

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DepthEstimation'))
from ground_plane_ransac import estimateGroundPlane, getCameraIntrinsics

TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'SampleData',
    'driverless-10fps',
    'frame_0001.jpg'
)

class TestHelper:
    def __init__(self, imagePath=TEST_IMAGE_PATH):
        self.imagePath = imagePath

    def loadTestImage(self):
        if not os.path.exists(self.imagePath):
            print(f"Test image not found at {self.imagePath}")
            print("Available sample images:")
            sampleDir = os.path.join(os.path.dirname(__file__), '..', 'SampleData', 'driverless-10fps')
            if os.path.exists(sampleDir):
                for f in sorted(os.listdir(sampleDir))[:5]:
                    print(f"  {f}")
            return None

        image = cv2.imread(self.imagePath)
        if image is None:
            print(f"Failed to load image from {self.imagePath}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def displayFourTile(self, visualization, windowName='Pipeline Output'):
        displayVis = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imshow(windowName, displayVis)
        print(f"Displaying {windowName}. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def saveVisualization(self, visualization, outputPath='test_output.png'):
        outputVis = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(outputPath, outputVis)
        print(f"Visualization saved to {outputPath}")

    def printConeDistances(self, segResults, distances):
        print("\n" + "="*60)
        print("CONE DETECTION AND DISTANCE ESTIMATION RESULTS")
        print("="*60)

        if segResults['numDetections'] == 0:
            print("No cones detected.")
            return

        print(f"Total cones detected: {segResults['numDetections']}\n")

        for i in range(segResults['numDetections']):
            className = segResults['classes'][i]['name']
            confidence = segResults['confidences'][i]

            print(f"Cone {i+1}: {className}")
            print(f"  Confidence: {confidence:.3f}")

            if i < len(distances):
                dist = distances[i]
                if dist['distance'] is not None:
                    print(f"  Final Distance: {dist['distance']:.3f} m")
                else:
                    print(f"  Final Distance: N/A")

                print(f"  Method Breakdown:")
                if dist['bboxDistance'] is not None:
                    print(f"    Bbox Height:   {dist['bboxDistance']:.3f} m")
                else:
                    print(f"    Bbox Height:   N/A")

                if dist['depthMapDistance'] is not None:
                    print(f"    Depth Map:     {dist['depthMapDistance']:.3f} m")
                else:
                    print(f"    Depth Map:     N/A")

                if dist['groundPlaneDistance'] is not None:
                    print(f"    Ground Plane:  {dist['groundPlaneDistance']:.3f} m")
                else:
                    print(f"    Ground Plane:  N/A")

            print()

        print("="*60)

    def runSingleFrameTest(self):
        print("Initializing models...")
        depthEstimator = DepthEstimator()
        coneSegmentor = ConeSegmentor()
        distEstimator = DistanceEstimator()

        print(f"Loading test image from {self.imagePath}...")
        image = self.loadTestImage()

        if image is None:
            return

        print("Running depth estimation...")
        depthMap = depthEstimator.estimateDepth(image)

        print("Running cone segmentation...")
        segResults = coneSegmentor.segment(image)

        H, W = image.shape[:2]
        cameraIntrinsics = getCameraIntrinsics(W, H, fov=90)

        print("Estimating ground plane...")
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

        print("Estimating cone distances...")
        coneDistances = distEstimator.estimateAllCones(
            segResults['boxes'],
            segResults['classes'],
            depthMap,
            planeParams
        )

        self.printConeDistances(segResults, coneDistances)

        print("Creating visualization...")
        visualization = createFourTileVisualization(
            image,
            depthMap,
            segResults,
            planeParams,
            inlierMask,
            coneDistances,
            cameraIntrinsics,
            depthScale=10.0
        )

        outputDir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(outputDir, exist_ok=True)

        outputPath = os.path.join(outputDir, 'test_visualization.png')
        self.saveVisualization(visualization, outputPath)

        self.displayFourTile(visualization)

def quickTest():
    helper = TestHelper()
    helper.runSingleFrameTest()

if __name__ == '__main__':
    quickTest()
