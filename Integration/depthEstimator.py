import os
import sys
import numpy as np
import cv2
import torch

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DepthEstimation', 'Depth-Anything-3', 'src'))

from depth_anything_3.api import DepthAnything3

class DepthEstimator:
    # fov and imageWidth are used to compute focalLength for metric depth scaling.
    def __init__(self, device='auto', fov=None, imageWidth=None, focalLength=None):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        if focalLength is not None:
            self.focalLength = focalLength
        elif fov is not None and imageWidth is not None:
            self.focalLength = imageWidth / (2 * np.tan(np.radians(fov) / 2))
        else:
            self.focalLength = None

        self.model = DepthAnything3.from_pretrained('depth-anything/DA3-SMALL')
        self.model.device = self.device
        self.model = self.model.to(self.device)
        self.model.eval()

    def estimateDepth(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel RGB image (H, W, 3)")

        H, W = image.shape[:2]

        processRes = 504
        with torch.no_grad():
            prediction = self.model.inference([image], process_res=processRes)

        depth = prediction.depth[0]

        if prediction.is_metric:
            metricDepth = depth
        elif self.focalLength is not None and prediction.intrinsics is not None:
            # Scale relative depth: metric_depth = depth * (actual_fx / predicted_fx)
            # Both focal lengths at the same (processed) resolution.
            procW = processRes if W >= H else round(W * processRes / H)
            actualFxProc = self.focalLength * procW / W
            predictedFxProc = float(prediction.intrinsics[0, 0, 0])
            if predictedFxProc > 1.0:
                metricDepth = depth * actualFxProc / predictedFxProc
            else:
                print("Warning: DA3 predicted degenerate focal length; returning relative depth")
                metricDepth = depth
        else:
            print("Warning: cannot compute metric depth (no intrinsics or focal length); returning relative depth")
            metricDepth = depth

        if metricDepth.shape != (H, W):
            metricDepth = cv2.resize(metricDepth, (W, H), interpolation=cv2.INTER_LINEAR)

        return metricDepth.astype(np.float32)
