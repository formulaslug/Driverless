## Validation / Code Safety

Functions that validate their inputs and outputs using checks:

- `imu_integration.py`
  - `load_imu_data`: checks that `csv_path` is a non-empty string, that all
    four required columns are present, and that the loaded DataFrame is non-empty.
  - `estimate_pose_from_imu`: checks required input columns (`time_ms`, `ax_mg`,
    `ay_mg`, `gyro_z_mdps`), non-empty input, and non-empty output.

- `yolact_integration.py`
  - `normalize_yolact_dict`: type-checks that the input is a `dict`, normalises
    key aliases (`scores` -> `confidence_scores`, `boxes` -> `bounding_boxes`),
    converts all arrays through `_to_numpy` to handle plain arrays and PyTorch
    tensors uniformly.
  - `parse_yolact_detections`: checks presence of `classes`, `confidence_scores`,
    and `bounding_boxes`, validates `bounding_boxes` shape is `(N, 4)`, verifies
    all three arrays have the same length, returns a typed empty DataFrame when
    `classes` is empty rather than raising.
  - `filter_low_confidence_detections`: checks `'confidence'` column exists and
    that `min_confidence ∈ [0, 1]`.
  - `attach_depth`: checks required columns (`pixel_u`, `pixel_v`, `mask_index`),
    validates `depth_map` is 2-D, performs explicit pixel bounds checking before
    any array indexing (raises with the offending pixel coordinates), and verifies
    `'depth_m'` is present in the output.
  - `process_frame` : validates `depth_map` is a 2-D NumPy array, returns a typed
    empty DataFrame when all detections fall below the confidence threshold,
    checks that `camera_transform.transform_detections` produced `x_vehicle` ,
    `y_vehicle`, `z_vehicle`.
  - `get_cone_measurements_for_kalman`: checks required columns (`x_vehicle`,
    `y_vehicle`, `cone_type`) and validates the locations array has exactly 2
    columns.

- `camera_transform.py`
  - `__init__`: raises if `camera_intrinsics` is `None`, if any of `fx`, `fy`,
    `cx`, `cy` are missing from the dict, or if `fx` or `fy` is zero.
  - `pixel_to_camera_frame`: rejects negative depth values.
  - `transform_detections`: checks `pixel_u`, `pixel_v`, `depth_m` exist in the
    input DataFrame and that `x_vehicle`, `y_vehicle`, `z_vehicle` are present
    in the output.

- `synchronize.py`
  - `__init__`: enforces `min_confidence ∈ [0, 1]`.
  - `add_yolact_result`: type-checks `frame_id` as `int` and `yolact_output`
    as `dict`.
  - `add_depth_map`: type-checks `frame_id` as `int`, validates `depth_map` is
    a 2-D NumPy array.
  - `_process_complete_frame`: checks that `get_cone_measurements_for_kalman`
    returns a `list `.

- `fusion_node.py`
  - `EgoMotionEstimator.__init__`: requires  a nonempty Polars DataFrame, verifies
    `estimate_pose_from_imu` output contains `ds`, `time_s`, and `theta`, raises
    if the resulting time array is empty.
  - `EgoMotionEstimator.get_motion_between`: type checks both `t_start` and
    `t_end` as numeric.
  - `MappingAndFusion.__init__`: type checks `ego_motion` as an
    `EgoMotionEstimator` instance, raises if `cone_filter` is `None`.
  - `MappingAndFusion.handle_perception_frame`: checks that `time_s` and
    `cone_measurements` keys exist in the frame dict and that
    `cone_measurements` is a `list`.
  - `MappingAndFusion.get_cone_map_for_planner`: checks that `getConeMap()`
    returns a `list`.
  - `PerceptionFusionPipeline.__init__`: raises if either `perception_sync` or
    `fusion` is `None`.