import json
import os
import shutil
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools import mask as cocoMask

def decodeUncompressedRLE(rleList, imageHeight, imageWidth):
    """
    Decode uncompressed RLE format to binary mask
    Format: [start_index, run_length, start_index, run_length, ...]
    Where start_index = y * width + x (flattened coordinates)
    """
    mask = np.zeros((imageHeight, imageWidth), dtype=np.uint8)

    for i in range(0, len(rleList), 2):
        if i + 1 >= len(rleList):
            break

        startIndex = int(rleList[i])
        runLength = int(rleList[i + 1])

        for j in range(runLength):
            index = startIndex + j
            if index >= imageHeight * imageWidth:
                continue

            y = index // imageWidth
            x = index % imageWidth

            if 0 <= y < imageHeight and 0 <= x < imageWidth:
                mask[y, x] = 1

    return mask

def isUncompressedRLE(poly, imageWidth, imageHeight):
    """
    Detect if polygon is actually uncompressed RLE format
    RLE has very large values (flattened indices) alternating with small values (run lengths)
    """
    if len(poly) < 4:
        return False

    maxVal = max(poly)
    threshold = max(imageWidth, imageHeight) * 2

    if maxVal <= threshold:
        return False

    evenVals = [poly[i] for i in range(0, len(poly), 2)]
    oddVals = [poly[i] for i in range(1, len(poly), 2)]

    avgEven = sum(evenVals) / len(evenVals) if evenVals else 0
    avgOdd = sum(oddVals) / len(oddVals) if oddVals else 0

    return avgEven > threshold and avgOdd < max(imageWidth, imageHeight)

def decodeCocoSegmentation(segmentation, imageHeight, imageWidth):
    """
    Decode COCO segmentation to polygon coordinates
    Handles RLE (compressed and uncompressed) and polygon formats
    """
    if isinstance(segmentation, dict):
        if 'counts' in segmentation:
            mask = cocoMask.decode(segmentation)
        else:
            return []
    elif isinstance(segmentation, list):
        if len(segmentation) == 0:
            return []

        polygons = []

        if isinstance(segmentation[0], list):
            for poly in segmentation:
                if len(poly) < 6:
                    continue

                if len(poly) % 2 != 0:
                    poly = poly[:-1]

                if isUncompressedRLE(poly, imageWidth, imageHeight):
                    mask = decodeUncompressedRLE(poly, imageHeight, imageWidth)
                elif max(poly) < max(imageWidth, imageHeight) * 1.5:
                    polygons.append(poly)
                    continue
                else:
                    try:
                        rle = cocoMask.frPyObjects([poly], imageHeight, imageWidth)
                        mask = cocoMask.decode(rle)
                    except:
                        continue

                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) >= 3:
                        epsilon = 0.002 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)

                        if len(approx) >= 3:
                            polygon = approx.reshape(-1).astype(float).tolist()
                            if len(polygon) >= 6:
                                polygons.append(polygon)

            return polygons
        else:
            if len(segmentation) % 2 != 0:
                segmentation = segmentation[:-1]

            if isUncompressedRLE(segmentation, imageWidth, imageHeight):
                mask = decodeUncompressedRLE(segmentation, imageHeight, imageWidth)
            elif max(segmentation) < max(imageWidth, imageHeight) * 1.5:
                return [segmentation]
            else:
                try:
                    rle = cocoMask.frPyObjects([segmentation], imageHeight, imageWidth)
                    mask = cocoMask.decode(rle)
                except:
                    return []
    else:
        return []

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) >= 3:
                polygon = approx.reshape(-1).astype(float).tolist()
                if len(polygon) >= 6:
                    polygons.append(polygon)

    return polygons

def convertCocoToYolo(cocoJsonPath, imagesRootPath, outputPath, trainRatio=0.8, valRatio=0.1):
    """
    Convert COCO segmentation format to YOLO segmentation format
    """
    print(f"Loading COCO annotations from {cocoJsonPath}")
    with open(cocoJsonPath, 'r') as f:
        cocoData = json.load(f)

    categories = {cat['id']: idx for idx, cat in enumerate(cocoData['categories'])}
    categoryNames = [cat['name'] for cat in sorted(cocoData['categories'], key=lambda x: x['id'])]

    print(f"Found {len(categories)} classes: {categoryNames}")

    imageIdToFilename = {img['id']: img['file_name'] for img in cocoData['images']}
    imageIdToDimensions = {img['id']: (img['width'], img['height']) for img in cocoData['images']}

    imageAnnotations = {}
    for ann in cocoData['annotations']:
        imageId = ann['image_id']
        if imageId not in imageAnnotations:
            imageAnnotations[imageId] = []
        imageAnnotations[imageId].append(ann)

    print(f"Processing {len(imageAnnotations)} images with annotations")

    outputPath = Path(outputPath)
    trainImgPath = outputPath / 'images' / 'train'
    valImgPath = outputPath / 'images' / 'val'
    testImgPath = outputPath / 'images' / 'test'
    trainLblPath = outputPath / 'labels' / 'train'
    valLblPath = outputPath / 'labels' / 'val'
    testLblPath = outputPath / 'labels' / 'test'

    for path in [trainImgPath, valImgPath, testImgPath, trainLblPath, valLblPath, testLblPath]:
        path.mkdir(parents=True, exist_ok=True)

    allImageIds = list(imageAnnotations.keys())
    np.random.seed(42)
    np.random.shuffle(allImageIds)

    trainSplit = int(len(allImageIds) * trainRatio)
    valSplit = int(len(allImageIds) * (trainRatio + valRatio))

    trainIds = allImageIds[:trainSplit]
    valIds = allImageIds[trainSplit:valSplit]
    testIds = allImageIds[valSplit:]

    print(f"Split: Train={len(trainIds)}, Val={len(valIds)}, Test={len(testIds)}")

    def processSplit(imageIds, imgPath, lblPath, splitName):
        print(f"\nProcessing {splitName} split...")
        skippedCount = 0
        totalPolygons = 0
        failedDecodes = 0

        for imageId in tqdm(imageIds, desc=f"Converting {splitName}"):
            filename = imageIdToFilename[imageId]
            width, height = imageIdToDimensions[imageId]

            sourcePath = Path(imagesRootPath) / filename
            if not sourcePath.exists():
                skippedCount += 1
                continue

            stem = Path(filename).stem
            targetImgPath = imgPath / Path(filename).name
            shutil.copy(sourcePath, targetImgPath)

            yoloAnnotations = []
            for ann in imageAnnotations[imageId]:
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue

                classId = categories[ann['category_id']]
                segmentation = ann['segmentation']

                try:
                    polygons = decodeCocoSegmentation(segmentation, height, width)
                except Exception as e:
                    failedDecodes += 1
                    continue

                for polygon in polygons:
                    if len(polygon) < 6:
                        continue

                    normalizedPolygon = []
                    for i in range(0, len(polygon), 2):
                        x = max(0.0, min(1.0, polygon[i] / width))
                        y = max(0.0, min(1.0, polygon[i + 1] / height))
                        normalizedPolygon.extend([x, y])

                    if len(normalizedPolygon) >= 6:
                        polyStr = ' '.join(f'{coord:.6f}' for coord in normalizedPolygon)
                        yoloAnnotations.append(f"{classId} {polyStr}")
                        totalPolygons += 1

            if yoloAnnotations:
                labelPath = lblPath / f"{stem}.txt"
                with open(labelPath, 'w') as f:
                    f.write('\n'.join(yoloAnnotations))

        print(f"Converted {totalPolygons} polygons")
        if failedDecodes > 0:
            print(f"Failed to decode {failedDecodes} annotations")
        if skippedCount > 0:
            print(f"Skipped {skippedCount} images (not found)")

    processSplit(trainIds, trainImgPath, trainLblPath, 'train')
    processSplit(valIds, valImgPath, valLblPath, 'val')
    processSplit(testIds, testImgPath, testLblPath, 'test')

    print(f"\nDataset conversion complete!")
    print(f"Output directory: {outputPath}")
    print(f"Classes: {categoryNames}")

    return categoryNames

def main():
    cocoJsonPath = '../Data/fsoco_segmentation_train/train_coco.json'
    imagesRootPath = '../Data/fsoco_segmentation_train'
    outputPath = './dataset'

    categoryNames = convertCocoToYolo(cocoJsonPath, imagesRootPath, outputPath)

    print(f"\nNext steps:")
    print(f"1. Verify dataset structure in {outputPath}")
    print(f"2. Run train.py to start training")

if __name__ == '__main__':
    main()
