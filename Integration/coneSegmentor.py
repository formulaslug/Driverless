import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Segmentation-YOLO'))

import torch
import numpy as np
from ultralytics import YOLO

class ConeSegmentor:
    def __init__(self, weightsPath=None, confThreshold=0.25, device='auto'):
        if weightsPath is None:
            weightsPath = os.path.join(
                os.path.dirname(__file__),
                '..',
                'Segmentation-YOLO',
                'runs',
                'segment',
                'cone-segmentation',
                'weights',
                'best.pt'
            )

        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.confThreshold = confThreshold
        self.model = YOLO(weightsPath)

    def segment(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel RGB image (H, W, 3)")

        results = self.model.predict(
            source=image,
            conf=self.confThreshold,
            save=False,
            device=self.device,
            verbose=False
        )

        result = results[0]

        output = {
            'masks': None,
            'boxes': None,
            'classes': None,
            'confidences': None,
            'numDetections': 0
        }

        if result.masks is not None:
            numDetections = len(result.masks)
            output['numDetections'] = numDetections
            output['masks'] = result.masks.data.cpu().numpy()
            output['boxes'] = result.boxes.xyxy.cpu().numpy()
            output['classes'] = [{
                'id': int(cls),
                'name': result.names[int(cls)]
            } for cls in result.boxes.cls]
            output['confidences'] = result.boxes.conf.cpu().numpy()

        return output
