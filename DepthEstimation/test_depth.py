#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import torch
torch.backends.mps.is_available = lambda: False

import matplotlib
from pathlib import Path

sys.path.insert(0, 'Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

def loadModel(encoder='vits', device='auto'):
    print("Loading Depth-Anything-V2 model...")

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            print("Note: Using CPU (MPS disabled to avoid compatibility issues)")

    print(f"Device: {device}")
    print(f"Encoder: {encoder}")

    modelConfigs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    model = DepthAnythingV2(**modelConfigs[encoder])

    checkpointPath = f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'
    if not os.path.exists(checkpointPath):
        print(f"ERROR: Checkpoint not found: {checkpointPath}")
        print("Please run the setup first to download model weights")
        sys.exit(1)

    model.load_state_dict(torch.load(checkpointPath, map_location='cpu'))
    model = model.to(device).eval()

    print("Model loaded successfully!\n")
    return model, device

def createVideoFromFrames(outputDir, outputFps, width, height):
    print("\nCreating MP4 video from depth frames...")

    import glob
    frameFiles = sorted(glob.glob(os.path.join(outputDir, "frame_*_depth.png")))

    if not frameFiles:
        print("ERROR: No depth frames found to create video")
        return None

    outputVideoPath = os.path.join(outputDir, "depth_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(outputVideoPath, fourcc, outputFps, (width, height))

    print(f"Writing {len(frameFiles)} frames to video...")
    for frameFile in frameFiles:
        frame = cv2.imread(frameFile)
        if frame is not None:
            videoWriter.write(frame)

    videoWriter.release()
    print(f"Video created: {outputVideoPath}")
    return outputVideoPath

def processVideo(model, videoPath, outputDir, fps=10.0, maxFrames=10, inputSize=518, device='cpu'):
    os.makedirs(outputDir, exist_ok=True)

    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {videoPath}")
        return

    videoFps = cap.get(cv2.CAP_PROP_FPS)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video Info:")
    print(f"  Path: {videoPath}")
    print(f"  Frames: {totalFrames} @ {videoFps:.2f} FPS")
    print(f"  Resolution: {width}x{height}")
    print(f"  Processing: {fps} FPS (every {int(videoFps/fps)} frames)")
    if maxFrames > 0:
        print(f"  Max frames to process: {maxFrames}")
    print()

    frameInterval = max(1, int(videoFps / fps))
    frameIdx = 0
    processedCount = 0

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    print("Processing frames...")
    print("-" * 80)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if maxFrames > 0 and processedCount >= maxFrames:
            print(f"\nReached max frames limit ({maxFrames})")
            break

        if frameIdx % frameInterval == 0:
            try:
                depth = model.infer_image(frame, inputSize)

                depthMin = depth.min()
                depthMax = depth.max()
                depthMean = depth.mean()

                print(f"Frame {frameIdx:06d} | Depth: min={depthMin:.3f}, max={depthMax:.3f}, mean={depthMean:.3f}")

                np.save(
                    os.path.join(outputDir, f"frame_{frameIdx:06d}_depth.npy"),
                    depth
                )

                depthNorm = (depth - depthMin) / (depthMax - depthMin + 1e-8) * 255.0
                depthNorm = depthNorm.astype(np.uint8)
                depthVis = (cmap(depthNorm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                cv2.imwrite(
                    os.path.join(outputDir, f"frame_{frameIdx:06d}_depth.png"),
                    depthVis
                )

                processedCount += 1

            except Exception as e:
                print(f"ERROR processing frame {frameIdx}: {e}")

        frameIdx += 1

    cap.release()

    print("-" * 80)
    print(f"\nProcessing complete!")
    print(f"  Processed: {processedCount} frames (from {frameIdx} total)")
    print(f"  Output directory: {outputDir}")
    print(f"\nFiles saved:")
    print(f"  - frame_XXXXXX_depth.npy (raw metric depth values)")
    print(f"  - frame_XXXXXX_depth.png (visualization)")

    videoPath = createVideoFromFrames(outputDir, fps, width, height)

    if videoPath:
        print(f"\n✓ MP4 video created: {videoPath}")
        print(f"  Video FPS: {fps}")
        print(f"  Total frames: {processedCount}")

def main():
    parser = argparse.ArgumentParser(
        description='Simple Depth Estimation Test Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10 frames)
  python test_depth.py

  # Process more frames
  python test_depth.py --max-frames 50

  # Process full video
  python test_depth.py --max-frames -1

  # Custom video and output
  python test_depth.py --video path/to/video.mp4 --output my_results --fps 2.0
        """
    )

    parser.add_argument('--video', type=str, default='TestData/test1.mp4',
                        help='Path to input video (default: TestData/test1.mp4)')
    parser.add_argument('--output', type=str, default='depth_output',
                        help='Output directory (default: depth_output)')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Frames per second to process (default: 10.0)')
    parser.add_argument('--max-frames', type=int, default=10,
                        help='Max frames to process, -1 for all (default: 10)')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'],
                        help='Model encoder size (default: vits=smallest/fastest)')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input size for model (default: 518)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto (default: auto)')

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    print("=" * 80)
    print("DEPTH ESTIMATION TEST SCRIPT")
    print("=" * 80)
    print()

    model, device = loadModel(encoder=args.encoder, device=args.device)

    processVideo(
        model=model,
        videoPath=args.video,
        outputDir=args.output,
        fps=args.fps,
        maxFrames=args.max_frames,
        inputSize=args.input_size,
        device=device
    )

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
