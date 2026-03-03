# YolactEdge Cone Detection — Jetson AGX Orin

Real-time cone instance segmentation using YolactEdge with TensorRT acceleration on Jetson AGX Orin.

## Setup

Follow [yolactedge_jetson_setup.md](yolactedge_jetson_setup.md) for full installation instructions.

## Quick Start

### 1. Place weights

Place `fsoco_4k_49_40000.pth` in `~/yolact_edge/weights/`

### 2. Live Video Inference

Place `live_inference.py` in `~/yolact_edge/`:

```bash
python3 live_inference.py --trained_model=weights/fsoco_4k_49_40000.pth \
  --use_fp16_tensorrt \
  --config fsoco_4k_config \
  --score_threshold 0.5 \
  --video 0
```

### Output

Each frame prints a detection dictionary:

```python
{
    "classes":  np.array([2, 1, 4]),                                  # class indices
    "names":    ["orange cone", "large orange cone", "yellow cone"],  # class names
    "scores":   np.array([0.97, 0.91, 0.85]),                         # confidence scores (0-1)
    "boxes":    np.array([[x1,y1,x2,y2], ...]),                       # bounding box coordinates 
    "masks":    np.array([...])                                       # binary segmentation masks (H x W per detection)
}
```

- `classes` — `np.array` (int) — Class index for each detection
- `names` — `list[str]` — Human-readable class name for each detection
- `scores` — `np.array` (float) — Confidence score for each detection
- `boxes` — `np.array` (float) — Bounding boxes as `[x1, y1, x2, y2]` in pixels
- `masks` — `np.array` (bool) — Per-detection binary segmentation mask at frame resolution

Detections below `--score_threshold` are filtered out. The `--top_k` flag limits the max number of detections per frame.
