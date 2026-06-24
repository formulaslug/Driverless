## Failure Modes and Return

- **IMU loading (`load_imu_data`)**
  - When it fails:
    - `csv_path` is not a non-empty string, the CSV is missing any of the four
      required columns (`General.Time (millis)`, `ISM330.Accel X (milli-g)`,
      `ISM330.Accel Y (milli-g)`, `ISM330.Gyro Z (milli-dps)`), or the file
      loads but contains no rows.
  - What happens:
    - Raises `ValueError` with a descriptive message. The IMU pipeline stops at
      that point.

- **IMU pose estimation (`estimate_pose_from_imu`)**
  - When it fails:
    - The input DataFrame is empty or missing any of the four required columns
      (`time_ms`, `ax_mg`, `ay_mg`, `gyro_z_mdps`), or the multi-pass
      integration produces an empty result.
  - What happens:
    - Raises `ValueError`. No odometry output is produced for that run.

- **YOLACT parsing (`parse_yolact_detections`)**
  - When it fails:
    - The output dict is missing `classes`, `confidence_scores`/`scores`, or
      `bounding_boxes`/`boxes`, bounding boxes are not shape `(N, 4)`, or
      lengths of `classes`, `scores`, and `boxes` do not match.
  - What happens:
    - Raises `ValueError`. If `classes` is an empty array, returns a typed empty
      DataFrame (no exception) so downstream code handles zero detections
      gracefully.

- **Confidence filtering (`filter_low_confidence_detections`)**
  - When it fails:
    - The `'confidence'` column is absent, or `min_confidence` is outside `[0, 1]`.
  - What happens:
    - Raises `ValueError`. No filtered detections are returned.

- **Depth attachment (`attach_depth`)**
  - When it fails:
    - Required columns (`pixel_u`, `pixel_v`, `mask_index`) are missing,
      `depth_map` is not 2-D, or any detection pixel falls outside the depth
      map bounds.
  - What happens:
    - Raises `ValueError` naming the out-of-bounds pixel. That frame produces
      no depth-annotated detections.

- **Full frame processing (`process_frame`)**
  - When it fails:
    - `depth_map` is not a 2-D NumPy array, or any sub-step raises.
    - `camera_transform.transform_detections` does not produce `x_vehicle`,
      `y_vehicle`, `z_vehicle`.
  - What happens:
    - Raises `ValueError`. If all detections are filtered by confidence, returns
      a typed empty DataFrame - no exception.

- **Camera transform (`CameraToVehicleTransform`)**
  - When it fails:
    - Intrinsics dict is missing any of `fx`, `fy`, `cx`, `cy`, `fx` or `fy`
      is zero, depth passed to `pixel_to_camera_frame` is negative, or
      `pixel_u`, `pixel_v`, `depth_m` columns are absent from the detections
      DataFrame.
  - What happens:
    - Raises `ValueError`. The frame that triggered the error is dropped.

- **Synchronization (`PerceptionSynchronizer`)**
  - Normal behavior:
    - YOLACT outputs and depth maps are buffered by `frame_id`. A complete pair
      is processed immediately, the other half is held until its partner arrives.
  - When it fails:
    - Only one stream ever arrives for a `frame_id`, `frame_id` is not an `int`,
      `yolact_output` is not a `dict`, `depth_map` is not a 2-D array, or
      `_process_complete_frame` raises a validation error.
  - What happens:
    - If the partner never arrives, `add_yolact_result` / `add_depth_map`
      returns `None`, the cone map is not updated for that frame and processing
      continues normally.
    - Type/shape errors raise `TypeError` or `ValueError` immediately.
    - If `_process_complete_frame` raises, that frame is dropped and later
      frames are unaffected.

- **Ego-motion estimation (`EgoMotionEstimator`)**
  - When it fails:
    - The IMU DataFrame is empty or missing `ds`, `time_s`, or `theta` after
      integration, `t_start` / `t_end` are not numeric.
  - What happens:
    - Raises `ValueError` or `TypeError`. Ego-motion is not provided for that
      interval, so `MappingAndFusion.handle_perception_frame` does not run.

- **Fusion (`MappingAndFusion`)**
  - When it fails:
    - The perception frame dict is missing `time_s` or `cone_measurements`,
      `cone_measurements` is not a list, or the dependency objects passed to
      `__init__` are of the wrong type.
  - What happens:
    - Raises `ValueError` or `TypeError`. `ConeFilter.update` is not called for
      that frame, the existing cone map is preserved unchanged.