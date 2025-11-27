#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import RANSACRegressor

def getCameraIntrinsics(width, height, fx=None, fy=None, cx=None, cy=None, fov=60):
    """
    Get camera intrinsic parameters.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        fx, fy, cx, cy: Manual intrinsic parameters (optional)
        fov: Field of view in degrees (used if fx/fy not provided)

    Returns:
        dict with 'fx', 'fy', 'cx', 'cy'
    """
    if fx is None or fy is None:
        focalLength = width / (2 * np.tan(np.radians(fov) / 2))
        fx = fy = focalLength

    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2

    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

def estimateGroundPlane(depthMap, cameraIntrinsics, subsampleStep=5, groundRatioStart=0.6,
                        inlierThreshold=0.1, maxTrials=100, verbose=False):
    """
    Extract ground plane from depth map using RANSAC.

    Args:
        depthMap: (H, W) depth predictions from Depth Anything V2
        cameraIntrinsics: dict with 'fx', 'fy', 'cx', 'cy'
        subsampleStep: Subsample every Nth pixel for speed (default: 5)
        groundRatioStart: Sample from this ratio of image height (default: 0.6 = bottom 40%)
        inlierThreshold: RANSAC inlier threshold in meters (default: 0.1 = 10cm)
        maxTrials: Maximum RANSAC iterations (default: 100)
        verbose: Print debug information

    Returns:
        planeParams: [a, b, c, d] where ax + by + cz + d = 0
        inlierRatio: Fraction of points that are inliers
        inlierMask: Boolean mask of inlier points in the point cloud
    """
    H, W = depthMap.shape
    fx = cameraIntrinsics['fx']
    fy = cameraIntrinsics['fy']
    cx = cameraIntrinsics['cx']
    cy = cameraIntrinsics['cy']

    groundRegion = depthMap[int(groundRatioStart * H):, :]

    points3d = []
    pointIndices = []
    vStart = int(groundRatioStart * H)

    for v in range(groundRegion.shape[0]):
        for u in range(0, groundRegion.shape[1], subsampleStep):
            d = groundRegion[v, u]
            if d > 0 and np.isfinite(d):
                x = (u - cx) * d / fx
                y = (v + vStart - cy) * d / fy
                z = d
                points3d.append([x, y, z])
                pointIndices.append((v + vStart, u))

    if len(points3d) < 3:
        if verbose:
            print("ERROR: Not enough valid depth points for RANSAC")
        return None, 0.0, None

    points3d = np.array(points3d)

    if verbose:
        print(f"Extracted {len(points3d)} 3D points from ground region")

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
        if verbose:
            print(f"ERROR: RANSAC fitting failed: {e}")
        return None, 0.0, None

    a, b = ransac.estimator_.coef_
    c = -1
    d = ransac.estimator_.intercept_

    norm = np.sqrt(a**2 + b**2 + c**2)
    planeParams = np.array([a, b, c, d]) / norm

    inlierRatio = ransac.inlier_mask_.sum() / len(points3d)

    if verbose:
        print(f"Plane equation: {planeParams[0]:.4f}x + {planeParams[1]:.4f}y + {planeParams[2]:.4f}z + {planeParams[3]:.4f} = 0")
        print(f"Inlier ratio: {inlierRatio:.2%} ({ransac.inlier_mask_.sum()}/{len(points3d)} points)")

    fullInlierMask = np.zeros(depthMap.shape, dtype=bool)
    for idx, isInlier in enumerate(ransac.inlier_mask_):
        v, u = pointIndices[idx]
        fullInlierMask[v, u] = isInlier

    return planeParams, inlierRatio, fullInlierMask

def createVisualization(depthMap, inlierMask, planeParams, inlierRatio):
    """
    Create side-by-side visualization of depth map and ground plane.

    Args:
        depthMap: (H, W) raw depth values
        inlierMask: (H, W) boolean mask of ground plane inliers
        planeParams: [a, b, c, d] plane equation
        inlierRatio: Fraction of inlier points

    Returns:
        visualization: (H, W*3) BGR image with three panels
    """
    H, W = depthMap.shape

    depthMin = depthMap.min()
    depthMax = depthMap.max()
    depthNorm = (depthMap - depthMin) / (depthMax - depthMin + 1e-8) * 255.0
    depthNorm = depthNorm.astype(np.uint8)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depthVis = (cmap(depthNorm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    maskVis = np.zeros((H, W, 3), dtype=np.uint8)
    maskVis[inlierMask] = [0, 255, 0]
    maskVis[~inlierMask] = [0, 0, 255]

    combinedVis = depthVis.copy()
    alpha = 0.5
    combinedVis[inlierMask] = (alpha * depthVis[inlierMask] + (1-alpha) * np.array([0, 255, 0])).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1

    cv2.putText(depthVis, "Original Depth", (10, 30), font, fontScale, (255, 255, 255), thickness)
    cv2.putText(maskVis, "Ground Plane Mask", (10, 30), font, fontScale, (255, 255, 255), thickness)
    cv2.putText(combinedVis, "Combined View", (10, 30), font, fontScale, (255, 255, 255), thickness)

    cv2.putText(combinedVis, f"Inliers: {inlierRatio:.1%}", (10, H-10), font, fontScale, (255, 255, 255), thickness)

    separator = np.ones((H, 20, 3), dtype=np.uint8) * 255

    visualization = cv2.hconcat([depthVis, separator, maskVis, separator, combinedVis])

    return visualization

def processDepthFile(depthPath, outputDir, cameraIntrinsics, args, verbose=False):
    """
    Process a single depth .npy file and generate ground plane estimation.

    Args:
        depthPath: Path to .npy depth file
        outputDir: Output directory for results
        cameraIntrinsics: Camera intrinsic parameters
        args: Command-line arguments
        verbose: Print progress information

    Returns:
        planeParams if successful, None otherwise
    """
    basename = Path(depthPath).stem

    if verbose:
        print(f"\nProcessing {basename}...")

    try:
        depthMap = np.load(depthPath)
    except Exception as e:
        print(f"ERROR loading {depthPath}: {e}")
        return None

    if verbose:
        print(f"  Depth map shape: {depthMap.shape}")
        print(f"  Depth range: [{depthMap.min():.3f}, {depthMap.max():.3f}]")

    planeParams, inlierRatio, inlierMask = estimateGroundPlane(
        depthMap,
        cameraIntrinsics,
        subsampleStep=args.subsample,
        groundRatioStart=args.ground_ratio,
        inlierThreshold=args.inlier_threshold,
        maxTrials=args.max_trials,
        verbose=verbose
    )

    if planeParams is None:
        print(f"  FAILED: Could not estimate ground plane")
        return None

    planeOutputPath = os.path.join(outputDir, f"{basename}_plane.npy")
    np.save(planeOutputPath, planeParams)

    visualization = createVisualization(depthMap, inlierMask, planeParams, inlierRatio)

    visOutputPath = os.path.join(outputDir, f"{basename}_ground_plane.png")
    cv2.imwrite(visOutputPath, visualization)

    if verbose:
        print(f"  Saved plane params: {planeOutputPath}")
        print(f"  Saved visualization: {visOutputPath}")

    return planeParams

def processDirectory(inputDir, outputDir, cameraIntrinsics, args):
    """
    Process all .npy depth files in a directory.

    Args:
        inputDir: Input directory containing .npy files
        outputDir: Output directory for results
        cameraIntrinsics: Camera intrinsic parameters
        args: Command-line arguments
    """
    import glob

    depthFiles = sorted(glob.glob(os.path.join(inputDir, "*_depth.npy")))

    if not depthFiles:
        print(f"ERROR: No depth files found in {inputDir}")
        print("Expected files matching pattern: *_depth.npy")
        return

    print(f"Found {len(depthFiles)} depth files")
    print("-" * 80)

    successCount = 0

    for i, depthPath in enumerate(depthFiles):
        planeParams = processDepthFile(
            depthPath,
            outputDir,
            cameraIntrinsics,
            args,
            verbose=args.verbose
        )

        if planeParams is not None:
            successCount += 1

        if not args.verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(depthFiles)} files...")

    print("-" * 80)
    print(f"\nComplete! Successfully processed {successCount}/{len(depthFiles)} files")
    print(f"Results saved to: {outputDir}")

def main():
    parser = argparse.ArgumentParser(
        description='Ground Plane Estimation using RANSAC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all depth files from depth_output/
  python ground_plane_ransac.py

  # Process specific frame
  python ground_plane_ransac.py --frame 42

  # Custom intrinsics
  python ground_plane_ransac.py --fx 500 --fy 500 --cx 320 --cy 240

  # Adjust RANSAC parameters
  python ground_plane_ransac.py --inlier-threshold 0.15 --max-trials 200
        """
    )

    parser.add_argument('--input', type=str, default='depth_output',
                        help='Input directory with .npy depth files (default: depth_output)')
    parser.add_argument('--output', type=str, default='ground_plane_output',
                        help='Output directory (default: ground_plane_output)')
    parser.add_argument('--frame', type=int, default=None,
                        help='Process specific frame number (optional, otherwise all)')

    parser.add_argument('--fx', type=float, default=None,
                        help='Camera focal length x (optional)')
    parser.add_argument('--fy', type=float, default=None,
                        help='Camera focal length y (optional)')
    parser.add_argument('--cx', type=float, default=None,
                        help='Camera principal point x (optional)')
    parser.add_argument('--cy', type=float, default=None,
                        help='Camera principal point y (optional)')
    parser.add_argument('--fov', type=float, default=60,
                        help='Field of view in degrees (default: 60, used if fx/fy not specified)')

    parser.add_argument('--subsample', type=int, default=5,
                        help='Subsample every Nth pixel (default: 5)')
    parser.add_argument('--ground-ratio', type=float, default=0.6,
                        help='Start ground sampling from this ratio of image height (default: 0.6)')
    parser.add_argument('--inlier-threshold', type=float, default=0.1,
                        help='RANSAC inlier threshold in meters (default: 0.1)')
    parser.add_argument('--max-trials', type=int, default=100,
                        help='Maximum RANSAC iterations (default: 100)')

    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 80)
    print("GROUND PLANE ESTIMATION (RANSAC)")
    print("=" * 80)
    print()

    sampleDepthFile = None
    import glob
    depthFiles = glob.glob(os.path.join(args.input, "*_depth.npy"))
    if depthFiles:
        sampleDepthFile = depthFiles[0]
        sampleDepth = np.load(sampleDepthFile)
        H, W = sampleDepth.shape
    else:
        print("ERROR: No depth files found to determine image dimensions")
        sys.exit(1)

    cameraIntrinsics = getCameraIntrinsics(
        W, H,
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
        fov=args.fov
    )

    print(f"Camera intrinsics:")
    print(f"  fx={cameraIntrinsics['fx']:.2f}, fy={cameraIntrinsics['fy']:.2f}")
    print(f"  cx={cameraIntrinsics['cx']:.2f}, cy={cameraIntrinsics['cy']:.2f}")
    print(f"\nRANSAC parameters:")
    print(f"  Inlier threshold: {args.inlier_threshold}m")
    print(f"  Max trials: {args.max_trials}")
    print(f"  Ground region: bottom {(1-args.ground_ratio)*100:.0f}% of image")
    print()

    if args.frame is not None:
        depthPath = os.path.join(args.input, f"frame_{args.frame:06d}_depth.npy")
        if not os.path.exists(depthPath):
            print(f"ERROR: Frame file not found: {depthPath}")
            sys.exit(1)

        processDepthFile(depthPath, args.output, cameraIntrinsics, args, verbose=True)
    else:
        processDirectory(args.input, args.output, cameraIntrinsics, args)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
