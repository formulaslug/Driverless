import threading
from motionEstimator import MotionEstimator

# IMU-backed estimator. The car will eventually publish IMU samples; feed them
# via addSample (safe from any thread) and this integrates them into inter-frame
# motion. No IMU data exists yet, so estimate() returns zero motion.
class ImuMotionEstimator(MotionEstimator):
    def __init__(self):
        self.pendingSamples = []
        self.lock = threading.Lock()

    def addSample(self, imuSample):
        with self.lock:
            self.pendingSamples.append(imuSample)

    def estimate(self, frameId):
        with self.lock:
            self.pendingSamples = []
        # Real strapdown integration of accel/gyro goes here once timestamps and biases are available.
        return 0.0, 0.0, 0.0
