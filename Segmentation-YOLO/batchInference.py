import torch
from ultralytics import YOLO
import argparse
from pathlib import Path
from tqdm import tqdm

def runBatchInference(weightsPath, sourceDir, confThreshold=0.25, saveResults=True, outputDir=None):
    """
    Run YOLOv8 segmentation inference on multiple images
    """
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading model from {weightsPath}")
    print(f"Using device: {device}")

    model = YOLO(weightsPath)

    sourcePath = Path(sourceDir)
    if not sourcePath.exists():
        print(f"Error: Source directory not found: {sourceDir}")
        return

    imageExtensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    imageFiles = [f for f in sourcePath.iterdir()
                  if f.is_file() and f.suffix.lower() in imageExtensions]

    if len(imageFiles) == 0:
        print(f"No images found in {sourceDir}")
        return

    print(f"\nFound {len(imageFiles)} images")
    print(f"Confidence threshold: {confThreshold}")

    totalDetections = 0
    classDetections = {}

    results = model.predict(
        source=str(sourcePath),
        conf=confThreshold,
        save=saveResults,
        device=device,
        verbose=False,
        stream=True
    )

    print("\nProcessing images...")
    for result in tqdm(results, total=len(imageFiles)):
        if result.masks is not None:
            numDetections = len(result.masks)
            totalDetections += numDetections

            for cls in result.boxes.cls:
                className = result.names[int(cls)]
                classDetections[className] = classDetections.get(className, 0) + 1

    print("\n" + "="*60)
    print("Batch Inference Complete!")
    print("="*60)
    print(f"\nTotal images processed: {len(imageFiles)}")
    print(f"Total detections: {totalDetections}")
    print(f"Average detections per image: {totalDetections/len(imageFiles):.2f}")

    if classDetections:
        print("\nDetections by class:")
        for className, count in sorted(classDetections.items(), key=lambda x: x[1], reverse=True):
            print(f"  {className}: {count}")

    if saveResults:
        print(f"\nResults saved to: runs/segment/predict")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Batch Segmentation Inference')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained weights (.pt file)')
    parser.add_argument('--source', type=str, default='dataset/images/test',
                       help='Path to directory containing images')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results')

    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: Weights file not found: {args.weights}")
        return

    runBatchInference(
        weightsPath=args.weights,
        sourceDir=args.source,
        confThreshold=args.conf,
        saveResults=args.save
    )

if __name__ == '__main__':
    main()
