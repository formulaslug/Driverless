import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from typing import Union, List

project_root = Path(__file__).resolve().parent / 'YOLO'
sys.path.insert(0, str(project_root))

from yolo import create_model, create_converter
from yolo.utils.model_utils import get_device
from omegaconf import OmegaConf

class ConeDetector:
    def __init__(
        self,
        weightsPath: str,
        dataConfig: str = 'datasets/cones.yaml',
        model: str = 'v9-t',
        device: str = '0',
        confThreshold: float = 0.25,
        iouThreshold: float = 0.45,
        imageSize: int = 640
    ):
        self.weightsPath = Path(weightsPath)
        self.confThreshold = confThreshold
        self.iouThreshold = iouThreshold
        self.imageSize = imageSize

        datasetConfig = OmegaConf.load(dataConfig)
        self.classNames = datasetConfig.class_list
        self.classNum = datasetConfig.class_num

        self.colors = [
            (255, 127, 0),
            (0, 127, 255),
            (255, 165, 0),
            (255, 255, 0),
        ]

        self.device, _ = get_device(device)

        print(f"Loading model from {self.weightsPath}...")

        modelConfig = OmegaConf.create({'name': model})
        self.model = create_model(modelConfig, class_num=self.classNum, weight_path=str(self.weightsPath))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.converter = create_converter(model, self.model, None, [imageSize, imageSize], self.device)

        print(f"Model loaded successfully. Ready for inference.")

    def preprocessImage(self, img: np.ndarray):
        imgHeight, imgWidth = img.shape[:2]

        scale = min(self.imageSize / imgHeight, self.imageSize / imgWidth)
        newWidth = int(imgWidth * scale)
        newHeight = int(imgHeight * scale)

        resizedImg = cv2.resize(img, (newWidth, newHeight))

        paddedImg = np.zeros((self.imageSize, self.imageSize, 3), dtype=np.uint8)
        paddedImg[:newHeight, :newWidth] = resizedImg

        imgTensor = torch.from_numpy(paddedImg).permute(2, 0, 1).float() / 255.0
        imgTensor = imgTensor.unsqueeze(0).to(self.device)

        return imgTensor, scale

    def postprocess(self, predictions, scale):
        boxes = []
        scores = []
        classIds = []

        for pred in predictions:
            if pred is None:
                continue

            for detection in pred:
                if len(detection) < 6:
                    continue

                x1, y1, x2, y2 = detection[:4]
                conf = detection[4]
                classId = int(detection[5])

                if conf < self.confThreshold:
                    continue

                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

                boxes.append([x1, y1, x2, y2])
                scores.append(float(conf))
                classIds.append(classId)

        return boxes, scores, classIds

    def drawDetections(self, img: np.ndarray, boxes: List, scores: List, classIds: List):
        annotatedImg = img.copy()

        for box, score, classId in zip(boxes, scores, classIds):
            x1, y1, x2, y2 = box
            color = self.colors[classId % len(self.colors)]
            className = self.classNames[classId]

            cv2.rectangle(annotatedImg, (x1, y1), (x2, y2), color, 2)

            label = f'{className}: {score:.2f}'
            labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                annotatedImg,
                (x1, y1 - labelSize[1] - 4),
                (x1 + labelSize[0], y1),
                color,
                -1
            )

            cv2.putText(
                annotatedImg,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return annotatedImg

    def detect(self, imgPath: Union[str, Path], savePath: str = None, showResult: bool = False):
        imgPath = Path(imgPath)

        if not imgPath.exists():
            print(f"Error: Image not found at {imgPath}")
            return None

        img = cv2.imread(str(imgPath))

        if img is None:
            print(f"Error: Could not read image at {imgPath}")
            return None

        imgTensor, scale = self.preprocessImage(img)

        with torch.no_grad():
            predictions = self.model(imgTensor)

        predictions = self.converter.get_bbox(predictions, self.confThreshold, self.iouThreshold)

        boxes, scores, classIds = self.postprocess(predictions, scale)

        print(f"\nDetected {len(boxes)} cones in {imgPath.name}:")
        for box, score, classId in zip(boxes, scores, classIds):
            className = self.classNames[classId]
            print(f"  {className}: {score:.2f} at [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")

        annotatedImg = self.drawDetections(img, boxes, scores, classIds)

        if savePath:
            savePath = Path(savePath)
            savePath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(savePath), annotatedImg)
            print(f"\nSaved annotated image to: {savePath}")

        if showResult:
            cv2.imshow('Cone Detection', annotatedImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotatedImg, boxes, scores, classIds

    def detectBatch(self, imgDir: Union[str, Path], outputDir: str = None):
        imgDir = Path(imgDir)

        if not imgDir.exists():
            print(f"Error: Directory not found at {imgDir}")
            return

        imgExts = ['.jpg', '.jpeg', '.png', '.bmp']
        imgPaths = [p for p in imgDir.iterdir() if p.suffix.lower() in imgExts]

        print(f"Found {len(imgPaths)} images in {imgDir}")

        if outputDir:
            outputDir = Path(outputDir)
            outputDir.mkdir(parents=True, exist_ok=True)

        totalDetections = 0

        for imgPath in imgPaths:
            savePath = None
            if outputDir:
                savePath = outputDir / imgPath.name

            result = self.detect(imgPath, savePath=savePath)

            if result:
                _, boxes, _, _ = result
                totalDetections += len(boxes)

        print(f"\n{'=' * 60}")
        print(f"Batch processing complete!")
        print(f"Total images: {len(imgPaths)}")
        print(f"Total detections: {totalDetections}")
        print(f"Average detections per image: {totalDetections / len(imgPaths):.2f}")
        print(f"{'=' * 60}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run inference with trained cone detection model')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights')
    parser.add_argument('--source', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--output', type=str, default='output', help='Output directory for annotated images')
    parser.add_argument('--data', type=str, default='datasets/cones.yaml', help='Path to dataset config')
    parser.add_argument('--model', type=str, default='v9-t', help='Model architecture')
    parser.add_argument('--device', type=str, default='0', help='Device (cuda device, i.e. 0 or cpu)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--show', action='store_true', help='Show results')

    args = parser.parse_args()

    detector = ConeDetector(
        weightsPath=args.weights,
        dataConfig=args.data,
        model=args.model,
        device=args.device,
        confThreshold=args.conf,
        iouThreshold=args.iou,
        imageSize=args.img_size
    )

    sourcePath = Path(args.source)

    if sourcePath.is_file():
        detector.detect(sourcePath, savePath=Path(args.output) / sourcePath.name, showResult=args.show)
    elif sourcePath.is_dir():
        detector.detectBatch(sourcePath, outputDir=args.output)
    else:
        print(f"Error: {sourcePath} is neither a file nor a directory")
