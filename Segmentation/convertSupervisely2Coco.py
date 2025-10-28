import json
import os
import glob
import base64
import zlib
import numpy as np
from pathlib import Path
import cv2
from io import BytesIO

def decodeSuperviselyBitmap(bitmapData, origin, imgHeight, imgWidth):
    data = base64.b64decode(bitmapData['data'])
    pngData = zlib.decompress(data)

    maskImg = cv2.imdecode(np.frombuffer(pngData, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    if maskImg is None:
        return None

    mask = (maskImg > 0).astype(np.uint8)

    fullMask = np.zeros((imgHeight, imgWidth), dtype=np.uint8)
    y1, x1 = origin[1], origin[0]
    y2, x2 = min(y1 + mask.shape[0], imgHeight), min(x1 + mask.shape[1], imgWidth)
    fullMask[y1:y2, x1:x2] = mask[:y2-y1, :x2-x1]
    return fullMask

def maskToRle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()

def maskToBbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)]

def convertSupervisely2Coco(dataRoot, outputPath):
    metaPath = os.path.join(dataRoot, 'meta.json')
    with open(metaPath, 'r') as f:
        meta = json.load(f)

    segClasses = [c for c in meta['classes'] if c['shape'] == 'bitmap']
    classIdToCocoId = {c['id']: idx + 1 for idx, c in enumerate(segClasses)}

    cocoData = {
        'images': [],
        'annotations': [],
        'categories': [{'id': idx + 1, 'name': c['title'], 'supercategory': 'cone'}
                       for idx, c in enumerate(segClasses)]
    }

    print(f"Found {len(segClasses)} segmentation classes:")
    for cat in cocoData['categories']:
        print(f"  {cat['id']}: {cat['name']}")

    teamDirs = [d for d in os.listdir(dataRoot) if os.path.isdir(os.path.join(dataRoot, d))]

    imageId = 0
    annotationId = 0

    for teamDir in sorted(teamDirs):
        annDir = os.path.join(dataRoot, teamDir, 'ann')
        imgDir = os.path.join(dataRoot, teamDir, 'img')

        if not os.path.exists(annDir) or not os.path.exists(imgDir):
            continue

        annFiles = glob.glob(os.path.join(annDir, '*.json'))
        print(f"\nProcessing {teamDir}: {len(annFiles)} images", flush=True)

        for idx, annFile in enumerate(sorted(annFiles)):
            if idx % 20 == 0:
                print(f"  {teamDir}: {idx}/{len(annFiles)}", flush=True)
            with open(annFile, 'r') as f:
                ann = json.load(f)

            imgFilename = os.path.basename(annFile).replace('.json', '')
            imgPath = os.path.join(teamDir, 'img', imgFilename)

            cocoData['images'].append({
                'id': imageId,
                'file_name': imgPath,
                'height': ann['size']['height'],
                'width': ann['size']['width']
            })

            for obj in ann['objects']:
                if obj['geometryType'] != 'bitmap':
                    continue

                classId = obj['classId']
                if classId not in classIdToCocoId:
                    continue

                cocoId = classIdToCocoId[classId]

                fullMask = decodeSuperviselyBitmap(obj['bitmap'], obj['bitmap']['origin'],
                                                  ann['size']['height'], ann['size']['width'])

                if fullMask is None:
                    continue

                bbox = maskToBbox(fullMask)
                area = int(np.sum(fullMask))

                if area == 0:
                    continue

                rle = maskToRle(fullMask)

                cocoData['annotations'].append({
                    'id': annotationId,
                    'image_id': imageId,
                    'category_id': cocoId,
                    'segmentation': [rle],
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': 0
                })
                annotationId += 1

            imageId += 1

    print(f"\n{'='*60}")
    print(f"Total images: {len(cocoData['images'])}")
    print(f"Total annotations: {len(cocoData['annotations'])}")

    with open(outputPath, 'w') as f:
        json.dump(cocoData, f)

    print(f"Saved to: {outputPath}")

if __name__ == '__main__':
    dataRoot = 'Data/fsoco_segmentation_train'
    outputPath = 'Data/fsoco_segmentation_train/train_coco.json'
    convertSupervisely2Coco(dataRoot, outputPath)
