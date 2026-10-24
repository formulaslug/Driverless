import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelinePaths  # noqa: F401

import numpy as np
import torch

# Segmentation backends share one contract: segment(bgrImage) -> dict with keys
# masks, boxes (N,4 xyxy), classes ([{id,name}]), confidences (N,), numDetections.
# Class names must match CLASS_TO_COLOR_INDEX (seg_blue_cone, seg_yellow_cone,
# seg_orange_cone, seg_large_orange_cone, seg_unknown_cone). YOLO is the default
# on both Mac and Jetson (TensorRT .engine when exported); YOLACT-Edge is optional.

_emptyOutput = {'masks': None, 'boxes': None, 'classes': None, 'confidences': None, 'numDetections': 0}


class YoloBackend:
    def __init__(self, weightsPath=None, confThreshold=0.25, device='auto', imgSize=416):
        from ultralytics import YOLO

        self.deviceStr = _resolveDeviceStr(device)
        if weightsPath is None:
            weightsDir = os.path.join(
                pipelinePaths.repoRoot, 'Segmentation-YOLO', 'runs', 'segment',
                'cone-segmentation', 'weights'
            )
            enginePath = os.path.join(weightsDir, 'best.engine')
            useEngine = self.deviceStr == 'cuda' and os.path.exists(enginePath)
            weightsPath = enginePath if useEngine else os.path.join(weightsDir, 'best.pt')
        self.model = YOLO(weightsPath, task='segment')
        self.confThreshold = confThreshold
        self.imgSize = imgSize

    def segment(self, bgrImage):
        result = self.model.predict(bgrImage, device=self.deviceStr, verbose=False, conf=self.confThreshold, imgsz=self.imgSize)[0]
        names = result.names

        if result.boxes is None or len(result.boxes) == 0:
            return dict(_emptyOutput)

        classIds = result.boxes.cls.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
        classes = [{'id': int(c), 'name': names[int(c)]} for c in classIds]

        return {
            'masks': None,
            'boxes': boxes,
            'classes': classes,
            'confidences': confidences,
            'numDetections': len(classIds),
        }


# Thin adapter over the Integration ConeSegmentor (YOLACT-Edge), which expects RGB input.
class YolactEdgeBackend:
    def __init__(self, weightsPath=None, confThreshold=0.25, device='auto'):
        from coneSegmentor import ConeSegmentor
        self.model = ConeSegmentor(weightsPath=weightsPath, confThreshold=confThreshold, device=device)

    def segment(self, bgrImage):
        return self.model.segment(bgrImage[:, :, ::-1].copy())


def _resolveDeviceStr(device):
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device


def createSegmentationBackend(name, **kwargs):
    backends = {
        'yolo': YoloBackend,
        'yolactedge': YolactEdgeBackend,
    }
    if name not in backends:
        raise ValueError(f"Unknown segmentation backend '{name}'. Options: {list(backends)}")
    return backends[name](**kwargs)
