import sys
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DepthEstimation', 'Depth-Anything-V2'))

import torch
import numpy as np
import cv2
from depth_anything_v2.dpt import DepthAnythingV2

class DepthEstimator:
    def __init__(self, device='auto', modelSize='vits'):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        modelConfigs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        if modelSize not in modelConfigs:
            raise ValueError(f"Model size must be one of {list(modelConfigs.keys())}")

        config = modelConfigs[modelSize]

        self.model = DepthAnythingV2(**config)

        modelFile = f'depth_anything_v2_{modelSize}.pth'
        modelPath = os.path.join(
            os.path.dirname(__file__),
            '..',
            'DepthEstimation',
            'Depth-Anything-V2',
            'checkpoints',
            modelFile
        )

        if not os.path.exists(modelPath):
            modelPath = f'checkpoints/{modelFile}'

        self.modelLoaded = False
        if os.path.exists(modelPath):
            try:
                self.model.load_state_dict(torch.load(modelPath, map_location='cpu'))
                self.modelLoaded = True
            except Exception as e:
                print(f"Warning: Failed to load depth model from {modelPath}: {e}")
        else:
            print(f"Warning: Depth model not found at {modelPath}, using uninitialized weights")

        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Warning: Failed to move model to {self.device}, falling back to CPU: {e}")
            self.device = "cpu"
            self.model = self.model.to(self.device)

        self.model.eval()

    def estimateDepth(self, image, inputSize=518):
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel RGB image (H, W, 3)")

        if not self.modelLoaded:
            print("Warning: Running depth estimation with uninitialized model weights")

        imageBgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            depth = self.model.infer_image(imageBgr, inputSize)

        return depth
