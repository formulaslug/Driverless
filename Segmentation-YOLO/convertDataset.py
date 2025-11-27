import json
import base64
import io
import zlib
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

class SuperviselyToYOLO:
    def __init__(self, sourceDir, outputDir, trainSplit=0.8):
        self.sourceDir = Path(sourceDir)
        self.outputDir = Path(outputDir)
        self.trainSplit = trainSplit

        self.classMapping = {
            'seg_orange_cone': 0,
            'seg_blue_cone': 1,
            'seg_large_orange_cone': 2,
            'seg_yellow_cone': 3
        }

        self.classNames = ['orange_cone', 'blue_cone', 'large_orange_cone', 'yellow_cone']

        self.trainDir = self.outputDir / 'train'
        self.valDir = self.outputDir / 'val'

    def decodeBitmap(self, bitmapData):
        bitmapBytes = base64.b64decode(bitmapData)
        decompressed = zlib.decompress(bitmapBytes)
        image = Image.open(io.BytesIO(decompressed))
        return np.array(image)

    def bitmapToBbox(self, bitmap, origin):
        rows, cols = np.where(bitmap > 0)

        if len(rows) == 0:
            return None

        yMin = rows.min() + origin[1]
        yMax = rows.max() + origin[1]
        xMin = cols.min() + origin[0]
        xMax = cols.max() + origin[0]

        return xMin, yMin, xMax, yMax

    def convertToYoloFormat(self, bbox, imgWidth, imgHeight):
        xMin, yMin, xMax, yMax = bbox

        xCenter = (xMin + xMax) / 2.0 / imgWidth
        yCenter = (yMin + yMax) / 2.0 / imgHeight
        width = (xMax - xMin) / imgWidth
        height = (yMax - yMin) / imgHeight

        xCenter = max(0, min(1, xCenter))
        yCenter = max(0, min(1, yCenter))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        return xCenter, yCenter, width, height

    def processAnnotation(self, annPath, imgPath):
        with open(annPath, 'r') as f:
            data = json.load(f)

        imgWidth = data['size']['width']
        imgHeight = data['size']['height']

        labels = []

        for obj in data['objects']:
            classTitle = obj.get('classTitle', '')

            if classTitle not in self.classMapping:
                continue

            classId = self.classMapping[classTitle]

            if obj['geometryType'] == 'bitmap':
                bitmapData = obj['bitmap']['data']
                origin = obj['bitmap']['origin']

                bitmap = self.decodeBitmap(bitmapData)
                bbox = self.bitmapToBbox(bitmap, origin)

                if bbox is None:
                    continue

                yoloBox = self.convertToYoloFormat(bbox, imgWidth, imgHeight)
                labels.append(f"{classId} {yoloBox[0]:.6f} {yoloBox[1]:.6f} {yoloBox[2]:.6f} {yoloBox[3]:.6f}")

        return labels

    def convert(self):
        print("Scanning dataset...")

        teamDirs = [d for d in self.sourceDir.iterdir() if d.is_dir()]

        allSamples = []
        for teamDir in teamDirs:
            annDir = teamDir / 'ann'
            imgDir = teamDir / 'img'

            if not annDir.exists() or not imgDir.exists():
                continue

            annFiles = list(annDir.glob('*.json'))

            for annFile in annFiles:
                imgName = annFile.stem

                imgFile = imgDir / imgName

                if not imgFile.exists():
                    possibleImgExts = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

                    for ext in possibleImgExts:
                        candidate = imgDir / (imgName + ext)
                        if candidate.exists():
                            imgFile = candidate
                            break

                if imgFile.exists():
                    allSamples.append((annFile, imgFile))

        print(f"Found {len(allSamples)} samples")

        random.shuffle(allSamples)

        splitIdx = int(len(allSamples) * self.trainSplit)
        trainSamples = allSamples[:splitIdx]
        valSamples = allSamples[splitIdx:]

        print(f"Train: {len(trainSamples)}, Validation: {len(valSamples)}")

        for split, samples in [('train', trainSamples), ('val', valSamples)]:
            imgOutDir = self.outputDir / split / 'images'
            labelOutDir = self.outputDir / split / 'labels'

            imgOutDir.mkdir(parents=True, exist_ok=True)
            labelOutDir.mkdir(parents=True, exist_ok=True)

            print(f"\nProcessing {split} split...")

            for annFile, imgFile in tqdm(samples):
                try:
                    labels = self.processAnnotation(annFile, imgFile)

                    if len(labels) == 0:
                        continue

                    newImgName = f"{annFile.parent.parent.name}_{imgFile.name}"
                    newLabelName = f"{annFile.parent.parent.name}_{imgFile.stem}.txt"

                    shutil.copy(imgFile, imgOutDir / newImgName)

                    with open(labelOutDir / newLabelName, 'w') as f:
                        f.write('\n'.join(labels))

                except Exception as e:
                    print(f"\nError processing {annFile}: {e}")
                    continue

        print("\nDataset conversion complete!")
        print(f"Output directory: {self.outputDir}")

        self.printStats()

    def printStats(self):
        print("\n=== Dataset Statistics ===")

        for split in ['train', 'val']:
            labelDir = self.outputDir / split / 'labels'

            if not labelDir.exists():
                continue

            labelFiles = list(labelDir.glob('*.txt'))

            classCount = {cls: 0 for cls in range(len(self.classNames))}
            totalBoxes = 0

            for labelFile in labelFiles:
                with open(labelFile, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            classId = int(parts[0])
                            classCount[classId] += 1
                            totalBoxes += 1

            print(f"\n{split.upper()}:")
            print(f"  Total images: {len(labelFiles)}")
            print(f"  Total bounding boxes: {totalBoxes}")
            for classId, count in classCount.items():
                className = self.classNames[classId]
                print(f"  {className}: {count}")

if __name__ == '__main__':
    sourceDir = '../Data/fsoco_segmentation_train'
    outputDir = 'datasets/cones'

    converter = SuperviselyToYOLO(sourceDir, outputDir, trainSplit=0.8)
    converter.convert()
