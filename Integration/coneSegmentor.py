import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Segmentation', 'yolact_edge'))

import torch
import torch.nn.functional as F
import numpy as np
from yolact_edge.data import cfg, set_cfg, MEANS, STD
from yolact_edge.yolact import Yolact
from yolact_edge.layers.output_utils import postprocess

class ConeSegmentor:
    def __init__(self, weightsPath=None, confThreshold=0.25, device='auto'):
        if weightsPath is None:
            weightsPath = os.path.join(
                os.path.dirname(__file__), '..', 'Segmentation', 'yolact_edge',
                'weights', 'yolact_edge_mobilenetv2_cone_2_750.pth'
            )

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.confThreshold = confThreshold

        set_cfg('yolact_edge_mobilenetv2_cone_config')
        self.classNames = cfg.dataset.class_names
        self.maxSize = cfg.max_size
        self.backboneTransform = cfg.backbone.transform
        self.mean = torch.Tensor(MEANS).float().to(self.device)[None, :, None, None]
        self.std = torch.Tensor(STD).float().to(self.device)[None, :, None, None]

        try:
            self.model = Yolact(training=False)
            self.model.load_weights(weightsPath)
            self.model.eval()
            self.model = self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLACT Edge model from {weightsPath}: {e}")

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

        output = {
            'masks': None,
            'boxes': None,
            'classes': None,
            'confidences': None,
            'numDetections': 0
        }

        if len(results) == 0:
            return output

        result = results[0]

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
