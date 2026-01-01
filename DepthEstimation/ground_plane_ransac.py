import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
import json

GOPRO_PRESETS = {
    'wide': {'fov': 118, 'description': 'GoPro Wide (SuperView) - 118 deg'},
    'linear': {'fov': 85, 'description': 'GoPro Linear - 85 deg'},
    'medium': {'fov': 95, 'description': 'GoPro Medium - 95 deg'},
    'narrow': {'fov': 65, 'description': 'GoPro Narrow - 65 deg'},
}
DEFAULT_GOPRO_MODE = 'wide'

def convertDisparityToDepth(disparity, scale=1.0):
    epsilon = 1e-6
    depth = scale / (disparity + epsilon)
    depth[disparity <= 0] = 0
    return depth

def getCameraIntrinsics(width, height, fx=None, fy=None, cx=None, cy=None, fov=None, goproMode=None):
    if goproMode is not None and goproMode in GOPRO_PRESETS:
        fov = GOPRO_PRESETS[goproMode]['fov']
    elif fov is None:
        fov = GOPRO_PRESETS[DEFAULT_GOPRO_MODE]['fov']

    if fov <= 0 or fov >= 180:
        raise ValueError(f"Invalid FOV {fov}: must be between 0 and 180 degrees")

    if fx is None or fy is None:
        focalLength = width / (2 * np.tan(np.radians(fov) / 2))
        fx = fy = focalLength

    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2

    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'fov': fov}

def loadExclusionAreas(jsonPath):
    try:
        with open(jsonPath, 'r') as f:
            exclusions = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Exclusion areas file not found: {jsonPath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in exclusion areas file {jsonPath}: {e}")
    return exclusions

def estimateGroundPlane(depthMap, cameraIntrinsics, exclusionAreas=None, subsampleStep=5,
                        inlierThreshold=0.1, maxTrials=100, maxTiltAngle=45,
                        depthScale=10.0, maxDepth=100.0):
    metricDepth = convertDisparityToDepth(depthMap, depthScale)

    H, W = metricDepth.shape
    fx = cameraIntrinsics['fx']
    fy = cameraIntrinsics['fy']
    cx = cameraIntrinsics['cx']
    cy = cameraIntrinsics['cy']

    if exclusionAreas is None:
        exclusionAreas = []

    def isInExclusionArea(u, v, exclusions):
        for box in exclusions:
            if box['x1'] <= u <= box['x2'] and box['y1'] <= v <= box['y2']:
                return True
        return False

    points3d = []
    pointIndices = []

    for v in range(0, H, subsampleStep):
        for u in range(0, W, subsampleStep):
            if isInExclusionArea(u, v, exclusionAreas):
                continue
            d = metricDepth[v, u]
            if d > 0 and np.isfinite(d) and d < maxDepth:
                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                z = d
                points3d.append([x, y, z])
                pointIndices.append((v, u))

    if len(points3d) < 3:
        return None, 0.0, None

    points3d = np.array(points3d)

    X = points3d[:, :2]
    y = points3d[:, 2]

    ransac = RANSACRegressor(
        residual_threshold=inlierThreshold,
        max_trials=maxTrials,
        random_state=42
    )

    try:
        ransac.fit(X, y)
    except Exception as e:
        return None, 0.0, None

    a, b = ransac.estimator_.coef_
    c = -1
    d = ransac.estimator_.intercept_

    norm = np.sqrt(a**2 + b**2 + c**2)
    if norm == 0:
        return None, 0.0, None
    planeParams = np.array([a, b, c, -d]) / norm

    normal = planeParams[:3]
    upVector = np.array([0, -1, 0])
    dotProduct = np.abs(np.dot(normal, upVector))
    tiltAngle = np.degrees(np.arccos(np.clip(dotProduct, 0, 1)))

    inlierRatio = ransac.inlier_mask_.sum() / len(points3d)

    fullInlierMask = np.zeros(depthMap.shape, dtype=bool)
    for idx, isInlier in enumerate(ransac.inlier_mask_):
        v, u = pointIndices[idx]
        fullInlierMask[v, u] = isInlier

    return planeParams, inlierRatio, fullInlierMask

def generatePlaneGrid(planeParams, cameraIntrinsics, depthMap, gridSpacing=2.0, depthScale=10.0, maxDepth=100.0):
    a, b, c, d = planeParams

    if abs(b) < 1e-6:
        return [], 0, 0

    metricDepth = convertDisparityToDepth(depthMap, depthScale)
    validDepths = metricDepth[(metricDepth > 0) & (metricDepth < maxDepth)]
    if len(validDepths) == 0:
        return [], 0, 0

    minZ = max(np.percentile(validDepths, 5), 0.5)
    maxZ = min(np.percentile(validDepths, 95), 50.0)

    fov = cameraIntrinsics.get('fov', 90)
    maxX = maxZ * np.tan(np.radians(fov / 2))

    zVals = np.arange(minZ, maxZ, gridSpacing)
    xVals = np.arange(-maxX, maxX, gridSpacing)

    gridPoints = []
    for x in xVals:
        row = []
        for z in zVals:
            y = (d - a * x - c * z) / b
            if -5 < y < 5:
                row.append([x, y, z])
            else:
                row.append(None)
        gridPoints.append(row)

    return gridPoints, len(xVals), len(zVals)

def projectPointsTo2D(points3d, cameraIntrinsics, imageShape):
    H, W = imageShape
    fx = cameraIntrinsics['fx']
    fy = cameraIntrinsics['fy']
    cx = cameraIntrinsics['cx']
    cy = cameraIntrinsics['cy']

    points2d = []
    validIndices = []

    for idx, (x, y, z) in enumerate(points3d):
        if z > 0:
            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)
            if 0 <= u < W and 0 <= v < H:
                points2d.append([u, v])
                validIndices.append(idx)

    return np.array(points2d), validIndices

def createVisualization(depthMap, inlierMask, planeParams, inlierRatio, originalFrame=None,
                        cameraIntrinsics=None, exclusionAreas=None, showInliers=True, depthScale=10.0):
    H, W = depthMap.shape

    depthMin = depthMap.min()
    depthMax = depthMap.max()
    depthNorm = (depthMap - depthMin) / (depthMax - depthMin + 1e-8) * 255.0
    depthNorm = depthNorm.astype(np.uint8)

    try:
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    except (KeyError, AttributeError):
        cmap = matplotlib.colormaps.get_cmap('viridis')
    depthVis = (cmap(depthNorm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    barWidth = 30
    barHeight = 200
    barX = W - barWidth - 20
    barY = 50

    for i in range(barHeight):
        colorVal = int(255 * (i / barHeight))
        colorRgb = cmap(colorVal)[:3]
        colorBgr = (int(colorRgb[2] * 255), int(colorRgb[1] * 255), int(colorRgb[0] * 255))
        cv2.rectangle(depthVis, (barX, barY + i), (barX + barWidth, barY + i + 1), colorBgr, -1)

    cv2.rectangle(depthVis, (barX, barY), (barX + barWidth, barY + barHeight), (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    thickness = 1

    numLabels = 5
    for i in range(numLabels):
        distance = depthMin + (depthMax - depthMin) * i / (numLabels - 1)
        labelY = barY + barHeight - int(barHeight * i / (numLabels - 1))
        label = f"{distance:.1f}m"
        textSize = cv2.getTextSize(label, font, fontScale, thickness)[0]
        cv2.putText(depthVis, label, (barX - textSize[0] - 5, labelY + 4),
                   font, fontScale, (255, 255, 255), thickness)

    if exclusionAreas is not None and len(exclusionAreas) > 0:
        for box in exclusionAreas:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            cv2.rectangle(depthVis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(depthVis, "EXCLUDED FROM RANSAC", (x1 + 5, y1 + 20), font, fontScale*0.8, (0, 0, 255), thickness)

    if originalFrame is not None:
        overlayVis = originalFrame.copy()
    else:
        overlayVis = depthVis.copy()

    if inlierMask is not None and showInliers:
        inlierOverlay = np.zeros_like(overlayVis)
        inlierOverlay[inlierMask] = (255, 0, 255)

        overlayVis = cv2.addWeighted(overlayVis, 1.0, inlierOverlay, 0.25, 0)

    if cameraIntrinsics is not None and planeParams is not None:
        gridPoints, numX, numZ = generatePlaneGrid(planeParams, cameraIntrinsics, depthMap, gridSpacing=2.0, depthScale=depthScale)
        if numX > 0 and numZ > 0:
            fx = cameraIntrinsics['fx']
            fy = cameraIntrinsics['fy']
            cx = cameraIntrinsics['cx']
            cy = cameraIntrinsics['cy']

            points2d = [[None for _ in range(numZ)] for _ in range(numX)]

            for xi in range(numX):
                for zi in range(numZ):
                    if gridPoints[xi][zi] is not None:
                        x, y, z = gridPoints[xi][zi]
                        if z > 0.1:
                            u = int(fx * x / z + cx)
                            v = int(fy * y / z + cy)
                            if -200 <= u < W + 200 and -200 <= v < H + 200:
                                points2d[xi][zi] = (u, v)

            wireframeColor = (0, 255, 0)
            wireframeThickness = 2

            for xi in range(numX):
                for zi in range(numZ - 1):
                    if points2d[xi][zi] and points2d[xi][zi + 1]:
                        pt1 = points2d[xi][zi]
                        pt2 = points2d[xi][zi + 1]
                        retval, p1, p2 = cv2.clipLine((0, 0, W, H), pt1, pt2)
                        if retval:
                            cv2.line(overlayVis, p1, p2, wireframeColor, wireframeThickness)

            for xi in range(numX - 1):
                for zi in range(numZ):
                    if points2d[xi][zi] and points2d[xi + 1][zi]:
                        pt1 = points2d[xi][zi]
                        pt2 = points2d[xi + 1][zi]
                        retval, p1, p2 = cv2.clipLine((0, 0, W, H), pt1, pt2)
                        if retval:
                            cv2.line(overlayVis, p1, p2, wireframeColor, wireframeThickness)

    a, b, c, d = planeParams

    separator = np.ones((H, 20, 3), dtype=np.uint8) * 255

    visualization = cv2.hconcat([depthVis, separator, overlayVis])

    return visualization

def processDepthFile(depthPath, outputDir, cameraIntrinsics, args):
    basename = Path(depthPath).stem
    try:
        depthMap = np.load(depthPath)
    except Exception as e:
        print(f"ERROR loading {depthPath}: {e}")
        return None

    planeParams, inlierRatio, inlierMask = estimateGroundPlane(
        depthMap,
        cameraIntrinsics,
        exclusionAreas=args.exclusion_areas,
        subsampleStep=args.subsample,
        inlierThreshold=args.inlier_threshold,
        maxTrials=args.max_trials,
        maxTiltAngle=args.max_tilt,
        depthScale=args.depth_scale,
    )

    if planeParams is None:
        return None

    originalFrame = None
    import re
    frameMatch = re.search(r'frame_(\d+)', basename)
    if frameMatch and os.path.exists(args.video):
        frameNum = int(frameMatch.group(1))
        cap = cv2.VideoCapture(args.video)
        if cap.isOpened():
            totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frameNum < totalFrames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
                ret, originalFrame = cap.read()
                if ret and originalFrame is not None:
                    originalFrame = cv2.resize(originalFrame, (depthMap.shape[1], depthMap.shape[0]))
            cap.release()

    planeOutputPath = os.path.join(outputDir, f"{basename}_plane.npy")
    np.save(planeOutputPath, planeParams)

    visualization = createVisualization(depthMap, inlierMask, planeParams, inlierRatio, originalFrame,
                                       cameraIntrinsics, args.exclusion_areas, args.show_inliers,
                                       args.depth_scale)

    visOutputPath = os.path.join(outputDir, f"{basename}_ground_plane.png")
    cv2.imwrite(visOutputPath, visualization)

    return planeParams

def processDirectory(inputDir, outputDir, cameraIntrinsics, args):
    import glob

    depthFiles = sorted(glob.glob(os.path.join(inputDir, "*_depth.npy")))

    successCount = 0

    for i, depthPath in enumerate(depthFiles):
        planeParams = processDepthFile(
            depthPath,
            outputDir,
            cameraIntrinsics,
            args,
        )

        if planeParams is not None:
            successCount += 1

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='depth_output',
                        help='Input directory with .npy depth files (default: depth_output)')
    parser.add_argument('--output', type=str, default='ground_plane_output',
                        help='Output directory (default: ground_plane_output)')
    parser.add_argument('--frame', type=int, default=None,
                        help='Process specific frame number (optional, otherwise all)')
    parser.add_argument('--video', type=str, default='TestData/test1.mp4',
                        help='Path to source video file for overlay visualization (default: TestData/test1.mp4)')

    parser.add_argument('--fx', type=float, default=None,
                        help='Camera focal length x (optional)')
    parser.add_argument('--fy', type=float, default=None,
                        help='Camera focal length y (optional)')
    parser.add_argument('--cx', type=float, default=None,
                        help='Camera principal point x (optional)')
    parser.add_argument('--cy', type=float, default=None,
                        help='Camera principal point y (optional)')
    parser.add_argument('--gopro-mode', type=str, default='wide',
                        choices=['wide', 'linear', 'medium', 'narrow'],
                        help='GoPro lens mode preset (default: wide=118deg)')
    parser.add_argument('--fov', type=float, default=None,
                        help='Custom FOV in degrees (overrides --gopro-mode if set)')

    parser.add_argument('--exclusions', type=str, required=True,
                        help='Path to JSON file with exclusion areas (format: [{"x1": int, "y1": int, "x2": int, "y2": int}])')
    parser.add_argument('--subsample', type=int, default=5,
                        help='Subsample every Nth pixel (default: 5)')
    parser.add_argument('--inlier-threshold', type=float, default=0.1,
                        help='RANSAC inlier threshold in meters (default: 0.1)')
    parser.add_argument('--max-trials', type=int, default=100,
                        help='Maximum RANSAC iterations (default: 100)')
    parser.add_argument('--max-tilt', type=float, default=45,
                        help='Maximum plane tilt from horizontal in degrees (default: 45)')
    parser.add_argument('--depth-scale', type=float, default=10.0,
                        help='Scale factor for disparity-to-depth conversion (default: 10.0)')

    parser.add_argument('--show-inliers', action='store_true', default=True,
                        help='Highlight inlier points in visualization (default: True)')

    args = parser.parse_args()

    args.exclusion_areas = loadExclusionAreas(args.exclusions)

    os.makedirs(args.output, exist_ok=True)

    sampleDepthFile = None
    import glob
    depthFiles = glob.glob(os.path.join(args.input, "*_depth.npy"))
    if depthFiles:
        sampleDepthFile = depthFiles[0]
        sampleDepth = np.load(sampleDepthFile)
        H, W = sampleDepth.shape

    cameraIntrinsics = getCameraIntrinsics(
        W, H,
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
        fov=args.fov,
        goproMode=args.gopro_mode if args.fov is None else None
    )

    if args.frame is not None:
        depthPath = os.path.join(args.input, f"frame_{args.frame:06d}_depth.npy")

        processDepthFile(depthPath, args.output, cameraIntrinsics, args)
    else:
        processDirectory(args.input, args.output, cameraIntrinsics, args)
    print("DONE!")

if __name__ == '__main__':
    main()
