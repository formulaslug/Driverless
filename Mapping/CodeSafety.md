## Validation / Code Safety

Functions that validate their inputs and outputs using checks:

- `imu_integration.py`
  - `load_imu_data`: checks CSV path string, required IMU columns, and non‑empty data.
  - `estimate_pose_from_imu`: checks required input columns and non‑empty output.

- `yolact_integration.py`
  - `normalize_yolact_dict`: type‑checks the input dict.
  - `parse_yolact_detections`: checks presence and shapes of `classes`, `scores`,
    `boxes`; ensures lengths match.
  - `filter_low_confidence_detections`: checks `'confidence'` column and that
    `min_confidence ∈ [0,1]`.
  - `attach_depth`: checks required columns, 2‑D `depth_map`, pixel bounds, and
    that `'depth_m'` is added
  - `process_frame`: checks 2‑D `depth_map` and that camera transform adds
    `x_vehicle`, `y_vehicle`, `z_vehicle`.
  - `get_cone_measurements_for_kalman`: checks required columns and array shape.

- `camera_transform.py`
  - `__init__`: checks intrinsics dict has `fx, fy, cx, cy` and `fx, fy ≠ 0`.
  - `pixel_to_camera_frame`: rejects negative depth
  - `transform_detections`: checks `pixel_u`, `pixel_v`, `depth_m` exist and
    that `x_vehicle`, `y_vehicle`, `z_vehicle` are produced.

- `fusion_node.py`
  - `EgoMotionEstimator.__init__`: requires non‑empty IMU DataFrame and checks
    odometry has `ds`, `time_s`, `theta`.
  - `get_motion_between`: type‑checks `t_start`, `t_end`.
  - `MappingAndFusion.__init__`: type‑checks dependencies.
  - `MappingAndFusion.handle_perception_frame`: checks for `time_s` and
    `cone_measurements` and that measurements is a list.
  - `PerceptionFusionPipeline` methods: type‑check `frame_id`, `yolact_output`,
    and 2‑D `depth_map`.

- `synchronize.py`
  - `__init__`: enforces `min_confidence ∈ [0,1]`.
  - `add_yolact_result` / `add_depth_map`: type‑check IDs and inputs.
  - `_process_complete_frame`: checks that cone measurements is a list and
    result dict has required keys.

- `se2_utils.py` / `se3_utils.py`
  - Type‑check numeric inputs and matrix/array shapes; raise `ValueError` on
    mismatches instead of silent failure.