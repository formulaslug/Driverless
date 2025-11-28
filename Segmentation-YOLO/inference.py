import torch
from ultralytics import YOLO
import argparse
import cv2
from pathlib import Path

def runInference(weightsPath, sourcePath, confThreshold=0.25, saveResults=True, showResults=False):
    """
    Run YOLOv8 segmentation inference on a single image
    """
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading model from {weightsPath}")
    print(f"Using device: {device}")

    model = YOLO(weightsPath)

    print(f"\nRunning inference on: {sourcePath}")
    print(f"Confidence threshold: {confThreshold}")

    results = model.predict(
        source=sourcePath,
        conf=confThreshold,
        save=saveResults,
        device=device,
        show=showResults,
        verbose=True
    )

    result = results[0]

    if result.masks is not None:
        numDetections = len(result.masks)
        print(f"\nDetected {numDetections} cones:")

        for i, (box, cls, conf, mask) in enumerate(zip(
            result.boxes.xyxy,
            result.boxes.cls,
            result.boxes.conf,
            result.masks.data
        )):
            className = result.names[int(cls)]
            print(f"  {i+1}. {className}: {conf:.2f}")

        if saveResults:
            outputPath = results[0].save_dir
            print(f"\nResults saved to: {outputPath}")
    else:
        print("\nNo detections found")

    return results

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation Inference')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results')
    parser.add_argument('--show', action='store_true',
                       help='Display results')

    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: Weights file not found: {args.weights}")
        return

    if not Path(args.source).exists():
        print(f"Error: Source image not found: {args.source}")
        return

    runInference(
        weightsPath=args.weights,
        sourcePath=args.source,
        confThreshold=args.conf,
        saveResults=args.save,
        showResults=args.show
    )

if __name__ == '__main__':
    main()
