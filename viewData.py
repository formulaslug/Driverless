import json
import base64
import io
import zlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

with open('Data/fsoco_segmentation_train/epflrt/ann/amz_00825.png.json') as f:
    ann = json.load(f)

img = Image.open('Data/fsoco_segmentation_train/epflrt/img/amz_00825.png')
fullMask = np.zeros((ann['size']['height'], ann['size']['width']), dtype=np.uint8)

for obj in ann['objects']:
    if obj['geometryType'] == 'bitmap':
        bitmapData = base64.b64decode(obj['bitmap']['data'])

        decompressed = zlib.decompress(bitmapData)
        mask = Image.open(io.BytesIO(decompressed)).convert('L')

        origin = obj['bitmap']['origin']
        maskArray = np.array(mask)
        fullMask[origin[1]:origin[1]+mask.height, origin[0]:origin[0]+mask.width] = np.maximum(fullMask[origin[1]:origin[1]+mask.height, origin[0]:origin[0]+mask.width], maskArray)

plt.figure(figsize=(15, 8))
plt.imshow(img)
plt.imshow(fullMask, alpha=0.5)
plt.show()
