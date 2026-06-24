## External Libraries Used

- **NumPy**
  - Used in: `camera_transform.py`, `yolact_integration.py`, `se2_utils.py`, `se3_utils.py`, `synchronize.py`, `fusion_node.py`.
  - Purpose: vector/matrix math, homogeneous coordinate transforms, `np.interp` for IMU time interpolation, `np.nonzero` for mask centroid computation, `np.median` for mask-based depth sampling.

- **Polars**
  - Used in: `imu_integration.py`, `yolact_integration.py`, `camera_transform.py`, `fusion_node.py`.
  - Purpose: CSV loading (`pl.read_csv`), column operations (`with_columns`, `select`, `filter`, `cum_sum`, `diff`, `fill_null`), storing and passing detection tables between pipeline stages, schema-enforced empty DataFrames for zero-detection frames.

- **PyTorch**
  - Tensors produced by the YOLACT model are converted to NumPy in `yolact_integration.py` via `_to_numpy`, which calls `.detach().cpu().numpy()` when a tensor is detected. The rest of the pipeline is pure NumPy/Polars and does not require PyTorch at runtime.a