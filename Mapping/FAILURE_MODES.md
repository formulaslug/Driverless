## Failure modes and return

- **IMU loading (`load_imu_data`, `estimate_pose_from_imu`)**
  - When it fails:
    - The CSV is missing required columns or is empty.
  - What happens right now:
    - We raise a `ValueError` and the current run of the IMU pipeline stops at that point.
    - Later this needs to be changed so we directly get IMU data from the team

- **YOLACT / depth processing (`process_frame`, `get_cone_measurements_for_kalman`)**
  - When it fails:
    - The YOLACT dict is malformed, lengths or shapes do not match, or `depth_map` is not a 2‑D array (Format Error).
  - What happens right now:
    - We raise a `ValueError` and that frame does not produce cone detections.

- **Synchronization (`PerceptionSynchronizer`)**
  - Normal behavior:
    - There are two buffers, one for YOLACT and depth.
    - We buffer YOLACT and depth by `frame_id` and only emit a frame when we have both.
  - When it fails:
    - Only one stream arrives for a `frame_id`, or `_process_complete_frame` raises a validation error.
  - What happens right now:
    - If only one stream arrives, `add_yolact_result` / `add_depth_map` returns `None`,
    we simply do not update the cone map for that frame and continue with future frames.
    - If `_process_complete_frame` raises, we drop that frame and continue processing later frames
    as usual.

- **Ego motion (`EgoMotionEstimator`)**
  - When it fails:
    - The IMU DataFrame is empty or missing required columns, or `get_motion_between` is called with invalid times.
  - What happens right now:
    - We raise an error and do not provide ego‑motion for that frame, so fusion for that frame does not run.

- **Fusion (`MappingAndFusion`)**
  - When it fails:
    - The perception frame dict is missing `time_s` or `cone_measurements`, or `cone_measurements` has the wrong type.
  - What happens right now:
    - We raise a `ValueError` and do not call `ConeFilter.update` for that frame.