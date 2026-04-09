## External Libraries Used

- **NumPy**
  - Used in: `camera_transform.py`, `yolact_integration.py`, `fusion_node.py`,
    `se2_utils.py`, `se3_utils.py`.
  - Purpose: vector math, interpolation (`np.interp`), trig (`np.sin`, `np.cos`),
    matrix multiplies, shape checks.

- **Polars**
  - Used in: `imu_integration.py`, `yolact_integration.py`, `camera_transform.py`,
    `fusion_node.py`.
  - Purpose: CSV loading (`pl.read_csv`), column ops (`with_columns`, `select`,
    `filter`, `cum_sum`, `diff`), storing detection tables.

- **PyTorch**
  - Used in the YOLACT inference code.
  - In `yolact_integration.py`, tensors are converted to NumPy via
    `.detach().cpu().numpy()` in `_to_numpy`.