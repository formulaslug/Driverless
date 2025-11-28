import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DepthEstimation', 'Prior-Depth-Anything'))

import torch
import numpy as np
from prior_depth_anything import PriorDepthAnything

class DepthEstimator:
    def __init__(self, device='auto', modelSize='vits'):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = PriorDepthAnything(
            device=self.device,
            frozen_model_size=modelSize,
            conditioned_model_size=modelSize
        )

    def estimateDepth(self, image, pattern=1000):
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel RGB image (H, W, 3)")

        depthTensor = self.model.infer_one_sample(
            image=image,
            prior=image,
            pattern=pattern,
            visualize=False
        )

        if isinstance(depthTensor, torch.Tensor):
            depthMap = depthTensor.cpu().numpy()
        else:
            depthMap = depthTensor

        return depthMap
