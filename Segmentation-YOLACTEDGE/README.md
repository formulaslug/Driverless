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

# External Calls

### PyTorch (`torch`, `torch.backends.cudnn`)
- `torch.from_numpy()` — converts numpy frame to tensor
- `torch.no_grad()` — disables gradient tracking during inference
- `torch.cuda.FloatTensor` / `torch.FloatTensor` — sets default tensor type
- `cudnn.benchmark`, `cudnn.fastest`, `cudnn.deterministic` — CUDA performance flags

### OpenCV (`cv2`)
- `cv2.VideoCapture()` — opens camera stream
- `vid.read()` — reads a frame from the camera
- `vid.get(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)` — gets camera resolution
- `vid.release()` — closes camera stream

### NumPy (`numpy`)
- `.cpu().numpy()` on tensors to convert detections to arrays

### yolact_edge
- `FastBaseTransform()` — resizes and normalizes frames before inference
- `net()` — runs the neural network inference
- `postprocess()` — converts raw network output to classes, scores, boxes, masks
- `convert_to_tensorrt()` — optimizes the model for Jetson using TensorRT
- `Yolact()` — instantiates the model
- `net.load_weights()` — loads trained weights from file

## Failure Modes

| Location | Failure | Returns |
|---|---|---|
| `cv2.VideoCapture()` | Camera not found or unavailable | Exits `live_camera()` early |
| `vid.read()` | There is no frame read | Breaks out of inference loop |
| `net()` | Model returns None or missing keys | Skips frame, increments frame counter |
| `postprocess()` | Returns None or wrong number of values(4 because of classes, scores, boxes, masks) | Skips frame, increments frame counter |
| `net.load_weights()` | Weights file missing or corrupt | Logs error, exits program |
