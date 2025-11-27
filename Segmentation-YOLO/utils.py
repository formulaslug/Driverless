import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt

def xywhToXyxy(x: float, y: float, w: float, h: float, imgWidth: int, imgHeight: int) -> Tuple[int, int, int, int]:
    centerX = x * imgWidth
    centerY = y * imgHeight
    width = w * imgWidth
    height = h * imgHeight

    x1 = int(centerX - width / 2)
    y1 = int(centerY - height / 2)
    x2 = int(centerX + width / 2)
    y2 = int(centerY + height / 2)

    return x1, y1, x2, y2

def xyxyToXywh(x1: int, y1: int, x2: int, y2: int, imgWidth: int, imgHeight: int) -> Tuple[float, float, float, float]:
    width = x2 - x1
    height = y2 - y1
    centerX = x1 + width / 2
    centerY = y1 + height / 2

    x = centerX / imgWidth
    y = centerY / imgHeight
    w = width / imgWidth
    h = height / imgHeight

    return x, y, w, h

def visualizeYoloLabel(imgPath: str, labelPath: str, classNames: List[str], savePath: str = None):
    img = cv2.imread(imgPath)

    if img is None:
        print(f"Error: Could not read image at {imgPath}")
        return None

    imgHeight, imgWidth = img.shape[:2]

    colors = [
        (255, 127, 0),
        (0, 127, 255),
        (255, 165, 0),
        (255, 255, 0),
    ]

    with open(labelPath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        classId = int(parts[0])
        x, y, w, h = map(float, parts[1:5])

        x1, y1, x2, y2 = xywhToXyxy(x, y, w, h, imgWidth, imgHeight)

        color = colors[classId % len(colors)]
        className = classNames[classId]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = className
        labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(
            img,
            (x1, y1 - labelSize[1] - 4),
            (x1 + labelSize[0], y1),
            color,
            -1
        )

        cv2.putText(
            img,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    if savePath:
        cv2.imwrite(savePath, img)
        print(f"Saved visualization to {savePath}")

    return img

def analyzeDataset(datasetDir: str, classNames: List[str]):
    datasetDir = Path(datasetDir)

    print("\n" + "=" * 60)
    print("Dataset Analysis")
    print("=" * 60)

    for split in ['train', 'val']:
        labelDir = datasetDir / split / 'labels'
        imgDir = datasetDir / split / 'images'

        if not labelDir.exists():
            print(f"\n{split.upper()}: Directory not found")
            continue

        labelFiles = list(labelDir.glob('*.txt'))
        imgFiles = list(imgDir.glob('*.[jp][pn]g')) if imgDir.exists() else []

        classCount = {i: 0 for i in range(len(classNames))}
        totalBoxes = 0
        boxSizes = []
        imgsWithCones = 0

        for labelFile in labelFiles:
            with open(labelFile, 'r') as f:
                lines = f.readlines()

            if len(lines) > 0:
                imgsWithCones += 1

            for line in lines:
                parts = line.strip().split()

                if len(parts) >= 5:
                    classId = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])

                    classCount[classId] += 1
                    totalBoxes += 1
                    boxSizes.append((w, h))

        print(f"\n{split.upper()}:")
        print(f"  Images: {len(imgFiles)}")
        print(f"  Label files: {len(labelFiles)}")
        print(f"  Images with cones: {imgsWithCones}")
        print(f"  Total bounding boxes: {totalBoxes}")

        if totalBoxes > 0:
            avgBoxesPerImg = totalBoxes / max(1, imgsWithCones)
            print(f"  Average boxes per image: {avgBoxesPerImg:.2f}")

        print(f"\n  Class distribution:")
        for classId, count in classCount.items():
            className = classNames[classId]
            percentage = (count / totalBoxes * 100) if totalBoxes > 0 else 0
            print(f"    {className}: {count} ({percentage:.1f}%)")

        if boxSizes:
            widths = [w for w, h in boxSizes]
            heights = [h for w, h in boxSizes]

            print(f"\n  Bounding box sizes (normalized):")
            print(f"    Width - Mean: {np.mean(widths):.4f}, Std: {np.std(widths):.4f}")
            print(f"    Height - Mean: {np.mean(heights):.4f}, Std: {np.std(heights):.4f}")

    print("\n" + "=" * 60)

def plotClassDistribution(datasetDir: str, classNames: List[str], savePath: str = None):
    datasetDir = Path(datasetDir)

    trainClassCount = {i: 0 for i in range(len(classNames))}
    valClassCount = {i: 0 for i in range(len(classNames))}

    for split, classCount in [('train', trainClassCount), ('val', valClassCount)]:
        labelDir = datasetDir / split / 'labels'

        if not labelDir.exists():
            continue

        labelFiles = list(labelDir.glob('*.txt'))

        for labelFile in labelFiles:
            with open(labelFile, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                if len(parts) >= 5:
                    classId = int(parts[0])
                    classCount[classId] += 1

    x = np.arange(len(classNames))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    trainCounts = [trainClassCount[i] for i in range(len(classNames))]
    valCounts = [valClassCount[i] for i in range(len(classNames))]

    ax.bar(x - width/2, trainCounts, width, label='Train', color='#3498db')
    ax.bar(x + width/2, valCounts, width, label='Validation', color='#e74c3c')

    ax.set_xlabel('Cone Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution in Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(classNames, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if savePath:
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution plot to {savePath}")

    plt.show()

    return fig

def validateDataset(datasetDir: str):
    datasetDir = Path(datasetDir)

    print("\n" + "=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    issues = []

    for split in ['train', 'val']:
        labelDir = datasetDir / split / 'labels'
        imgDir = datasetDir / split / 'images'

        if not labelDir.exists():
            issues.append(f"{split}/labels directory not found")
            continue

        if not imgDir.exists():
            issues.append(f"{split}/images directory not found")
            continue

        labelFiles = list(labelDir.glob('*.txt'))
        imgFiles = {f.stem: f for f in imgDir.glob('*.[jp][pn]g')}

        print(f"\n{split.upper()}:")

        missingImages = 0
        invalidLabels = 0

        for labelFile in labelFiles:
            stem = labelFile.stem

            if stem not in imgFiles:
                missingImages += 1
                issues.append(f"Missing image for label: {labelFile.name}")

            with open(labelFile, 'r') as f:
                lines = f.readlines()

            for lineNum, line in enumerate(lines, 1):
                parts = line.strip().split()

                if len(parts) < 5:
                    invalidLabels += 1
                    issues.append(f"Invalid label format in {labelFile.name} line {lineNum}")
                    continue

                try:
                    classId = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])

                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        invalidLabels += 1
                        issues.append(f"Out of range values in {labelFile.name} line {lineNum}")

                except ValueError:
                    invalidLabels += 1
                    issues.append(f"Invalid number format in {labelFile.name} line {lineNum}")

        print(f"  Total labels: {len(labelFiles)}")
        print(f"  Missing images: {missingImages}")
        print(f"  Invalid label lines: {invalidLabels}")

    print("\n" + "=" * 60)

    if issues:
        print(f"\nFound {len(issues)} issues")
        print("\nFirst 10 issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
    else:
        print("\nNo issues found! Dataset is valid.")

    print("=" * 60)

    return len(issues) == 0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset utilities for YOLO cone detection')
    parser.add_argument('--action', type=str, required=True, choices=['analyze', 'validate', 'visualize', 'plot'],
                        help='Action to perform')
    parser.add_argument('--dataset', type=str, default='datasets/cones', help='Path to dataset directory')
    parser.add_argument('--classes', type=str, nargs='+',
                        default=['orange_cone', 'blue_cone', 'large_orange_cone', 'yellow_cone'],
                        help='Class names')
    parser.add_argument('--image', type=str, help='Path to image (for visualize)')
    parser.add_argument('--label', type=str, help='Path to label (for visualize)')
    parser.add_argument('--output', type=str, help='Output path')

    args = parser.parse_args()

    if args.action == 'analyze':
        analyzeDataset(args.dataset, args.classes)
    elif args.action == 'validate':
        validateDataset(args.dataset)
    elif args.action == 'visualize':
        if not args.image or not args.label:
            print("Error: --image and --label required for visualize action")
        else:
            visualizeYoloLabel(args.image, args.label, args.classes, args.output)
    elif args.action == 'plot':
        plotClassDistribution(args.dataset, args.classes, args.output)
