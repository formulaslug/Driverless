# Motion estimators produce vehicle-frame inter-frame motion (dx forward,
# dy left, dyaw ccw) consumed by ConeFilter.update.
class MotionEstimator:
    def estimate(self, frameId):
        raise NotImplementedError


def createMotionEstimator(name, **kwargs):
    if name == 'imu':
        from imuMotionEstimator import ImuMotionEstimator
        return ImuMotionEstimator(**kwargs)
    if name == 'none':
        return _ZeroMotion()
    raise ValueError(f"Unknown motion estimator '{name}'. Options: imu, none")


class _ZeroMotion(MotionEstimator):
    def estimate(self, frameId):
        return 0.0, 0.0, 0.0
