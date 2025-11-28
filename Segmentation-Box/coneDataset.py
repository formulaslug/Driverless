import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import skimage.io
import skimage.color

class ConeDataset(Dataset):
    def __init__(self, rootDir, cocoJsonPath, transform=None):
        """
        Args:
            rootDir: Root directory containing the image folders
            cocoJsonPath: Path to the COCO format JSON annotation file
            transform: Optional transform to be applied on a sample
        """
        self.rootDir = rootDir
        self.transform = transform

        self.coco = COCO(cocoJsonPath)
        self.imageIds = self.coco.getImgIds()

        self.loadClasses()

    def loadClasses(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.cocoLabels = {}
        self.cocoLabelsInverse = {}

        for c in categories:
            self.cocoLabels[len(self.classes)] = c['id']
            self.cocoLabelsInverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.imageIds)

    def __getitem__(self, idx):
        img = self.loadImage(idx)
        annot = self.loadAnnotations(idx)
        sample = {'img': img, 'annot': annot}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def loadImage(self, imageIndex):
        imageInfo = self.coco.loadImgs(self.imageIds[imageIndex])[0]
        path = os.path.join(self.rootDir, imageInfo['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def loadAnnotations(self, imageIndex):
        annotationsIds = self.coco.getAnnIds(imgIds=self.imageIds[imageIndex], iscrowd=False)
        annotations = np.zeros((0, 5))

        if len(annotationsIds) == 0:
            return annotations

        cocoAnnotations = self.coco.loadAnns(annotationsIds)
        for idx, a in enumerate(cocoAnnotations):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.cocoLabelToLabel(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def cocoLabelToLabel(self, cocoLabel):
        return self.cocoLabelsInverse[cocoLabel]

    def labelToCocoLabel(self, label):
        return self.cocoLabels[label]

    def imageAspectRatio(self, imageIndex):
        image = self.coco.loadImgs(self.imageIds[imageIndex])[0]
        return float(image['width']) / float(image['height'])

    def numClasses(self):
        return len(self.classes)
