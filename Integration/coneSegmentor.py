import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Segmentation', 'yolact_edge'))

import torch
import torch.nn.functional as F
import numpy as np

# YOLACT-Edge's vendored cython NMS uses np.int, removed in numpy>=1.24.
if not hasattr(np, 'int'):
    np.int = int
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

        h, w = image.shape[:2]

        output = {
            'masks': None,
            'boxes': None,
            'classes': None,
            'confidences': None,
            'numDetections': 0
        }

        # Preprocessing (inline FastBaseTransform to avoid its hardcoded .cuda())
        bgrImage = image[:, :, ::-1].copy()
        frame = torch.from_numpy(bgrImage).to(self.device).float()
        img = frame.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, (self.maxSize, self.maxSize), mode='bilinear', align_corners=False)

        if self.backboneTransform.normalize:
            img = (img - self.mean) / self.std
        elif self.backboneTransform.subtract_means:
            img = img - self.mean
        elif self.backboneTransform.to_float:
            img = img / 255

        img = img[:, (2, 1, 0), :, :].contiguous()

        extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None}
        with torch.no_grad():
            preds = self.model(img, extras=extras)["pred_outs"]

        classes, scores, boxes, masks = postprocess(
            preds, w, h, crop_masks=True, score_threshold=self.confThreshold
        )

        if classes.nelement() == 0:
            return output

        numDetections = classes.size(0)
        output['numDetections'] = numDetections
        output['masks'] = masks.cpu().float().numpy()
        output['boxes'] = boxes.cpu().float().numpy()
        output['classes'] = [{'id': int(c), 'name': self.classNames[c]} for c in classes]
        output['confidences'] = scores.cpu().numpy()

        return output
