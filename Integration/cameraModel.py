import json
import os
import numpy as np

# Vehicle frame: x forward, y left, z up, origin on the ground below the vehicle
# reference point. Camera frame: X right, Y down, Z forward. cameraPitchDeg is
# positive when the camera looks down toward the road.
class CameraModel:
    def __init__(self, width, height, fx, fy, cx, cy, mountHeight, pitchRad, offsetX=0.0, offsetY=0.0):
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Invalid focal lengths fx={fx}, fy={fy}: must be positive")
        if mountHeight <= 0:
            raise ValueError(f"Invalid camera height {mountHeight}: must be positive")
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.mountHeight = mountHeight
        self.pitchRad = pitchRad
        self.offsetX = offsetX
        self.offsetY = offsetY

    @classmethod
    def fromConfig(cls, frameWidth, frameHeight, configPath=None, fov=None):
        if configPath is None:
            configPath = os.path.join(os.path.dirname(__file__), 'camera.json')
        try:
            with open(configPath, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Camera config not found: {configPath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in camera config {configPath}: {e}")

        scaleX = frameWidth / config['width']
        scaleY = frameHeight / config['height']
        hasPixelIntrinsics = all(config.get(k) is not None for k in ('fx', 'fy', 'cx', 'cy'))

        if fov is None and hasPixelIntrinsics:
            fx = config['fx'] * scaleX
            fy = config['fy'] * scaleY
            cx = config['cx'] * scaleX
            cy = config['cy'] * scaleY
        else:
            fovDeg = fov if fov is not None else config['fov']
            if fovDeg <= 0 or fovDeg >= 180:
                raise ValueError(f"Invalid FOV {fovDeg}: must be between 0 and 180 degrees")
            fx = fy = frameWidth / (2 * np.tan(np.radians(fovDeg) / 2))
            cx = frameWidth / 2
            cy = frameHeight / 2

        return cls(
            frameWidth, frameHeight, fx, fy, cx, cy,
            mountHeight=config['cameraHeight'],
            pitchRad=np.radians(config['cameraPitchDeg']),
            offsetX=config.get('cameraOffsetX', 0.0),
            offsetY=config.get('cameraOffsetY', 0.0),
        )

    # Vehicle-frame direction of the pixel ray, scaled so its camera-frame Z is 1.
    def rayDirection(self, u, v, pitchRad=None):
        pitch = self.pitchRad if pitchRad is None else pitchRad
        xNorm = (u - self.cx) / self.fx
        yNorm = (v - self.cy) / self.fy
        return np.array([
            np.cos(pitch) - yNorm * np.sin(pitch),
            -xNorm,
            -yNorm * np.cos(pitch) - np.sin(pitch),
        ])

    # Ground-plane (x, y) of the point at camera-frame depth zc along the pixel ray.
    def pointAtDepth(self, u, v, zc):
        direction = self.rayDirection(u, v)
        return self.offsetX + zc * direction[0], self.offsetY + zc * direction[1]

    # Intersection of the pixel ray with the flat road; None at or above the horizon.
    def groundPoint(self, u, v, pitchRad=None):
        direction = self.rayDirection(u, v, pitchRad)
        if direction[2] >= -1e-6:
            return None
        t = self.mountHeight / -direction[2]
        return self.offsetX + t * direction[0], self.offsetY + t * direction[1]
