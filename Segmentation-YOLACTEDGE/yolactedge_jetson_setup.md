# YolactEdge on Jetson AGX Orin (JetPack 5.1.2) — Setup Documentation

## System Info

- System: Jetson AGX Orin
- JetPack 5.1.2 (L4T R35.4.1)
- CUDA 11.4.315 (pre-installed) 
- cuDNN 8.6.0.166 (pre-installed)
- TensorRT 8.5.2.2 (pre-installed)
- PyTorch 2.1.0a0+41361538.nv23.06 (NVIDIA Jetson wheel) 
- torchvision 0.16.1+fdea156 (built from source) 
- Python 3.8 (locked to JetPack 5) 

## Step 1: Create Conda Environment

```bash
conda create -n yolactedge python=3.8 -y
conda activate yolactedge
```

## Step 2: Install PyTorch (NVIDIA Jetson wheel)

```bash
pip install https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

Standard `pip install torch` won't work (it installs x86 builds). Jetson needs NVIDIA's ARM aarch64 wheels. Wheel filename found via the [Seeed Studio wiki](https://wiki.seeedstudio.com/install_torch_on_recomputer/) and [NVIDIA PyTorch for Jetson docs](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

## Step 3: Install torchvision (build from source)

No pre-built wheel exists for JetPack 5 + PyTorch 2.1. Must compile from source. Version 0.16.1 matches PyTorch 2.1.

```bash
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
cd ~
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
```

Takes ~15 minutes to compile on the Orin. Make sure conda env is active first.

## Step 4: Install YolactEdge dependencies

```bash
pip install cython opencv-python pillow matplotlib
pip install "git+https://github.com/haotian-liu/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
pip install GitPython termcolor tensorboard
```

## Step 5: Clone YolactEdge

```bash
cd ~
git clone https://github.com/haotian-liu/yolact_edge.git
cd yolact_edge
```

## Step 6: Symlink TensorRT into conda env

JetPack installs TensorRT system-wide, but conda environments are isolated and can't see system packages. Fix with symlinks:

```bash
ln -s /usr/lib/python3.8/dist-packages/tensorrt $(python -c "import site; print(site.getsitepackages()[0])")/tensorrt
ln -s /usr/lib/python3.8/dist-packages/tensorrt-8.5.2.2.dist-info $(python -c "import site; print(site.getsitepackages()[0])")/tensorrt-8.5.2.2.dist-info
```

This is a common issue across all Jetson boards when using conda or virtualenvs. See [NVIDIA Docs](https://nvidia-jetson.piveral.com/jetson-orin-nano/using-tensorrt-in-a-conda-environment-on-jetson-orin-nano/).

## Step 7: Install torch2trt

```bash
cd ~/yolact_edge
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install --plugins
```

## Step 8: Verify everything

```bash
python3 -c "
import torch
import tensorrt
import torch2trt
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('TensorRT:', tensorrt.__version__)
print('torch2trt: OK')
"
```

Expected output:

```
PyTorch: 2.1.0a0+41361538.nv23.06
CUDA available: True
TensorRT: 8.5.2.2
torch2trt: OK
```

## Step 9: Add FSOCO config

Add to `~/yolact_edge/yolact_edge/data/config.py`:

```python
fsoco_4k_dataset = dataset_base.copy({
    'name': 'fsoco_4k',
    'train_images': './data/fsoco_4k/train/',
    'train_info': './data/fsoco_4k/train_annotations.coco.json',
    'valid_images': './data/fsoco_4k/valid/',
    'valid_info': './data/fsoco_4k/valid_annotations.coco.json',
    'has_gt': True,
    'class_names': ('blue cone', 'large orange cone', 'orange cone', 'other cone', 'yellow cone'),
    # Only name and class_names are needed for inference
})

fsoco_4k_config = yolact_edge_resnet50_config.copy({
    'name': 'fsoco_4k',
    'dataset': fsoco_4k_dataset,
    'num_classes': 6,
})
```

## Step 10: Transfer weights and run

Place `fsoco_4k_49_40000.pth` in `~/yolact_edge/weights/`


Live Video Inference:

-Place `live_inference.py` in `~/yolact_edge/`:

```bash
python3 live_inference.py --trained_model=weights/fsoco_4k_49_40000.pth \
  --use_fp16_tensorrt \
  --config fsoco_4k_config \
  --score_threshold 0.5 \
  --video 0 \
  --use_fp16_tensorrt
```

Single image inference:

```bash
python3 eval.py --trained_model=weights/fsoco_4k_49_40000.pth \
  --score_threshold=0.3 --top_k=100 \
  --image=input.jpg:output.jpg \
  --use_fp16_tensorrt
```

Benchmark FPS:

```bash
python3 eval.py --trained_model=weights/fsoco_4k_49_40000.pth \
  --use_fp16_tensorrt --benchmark --max_images=1000
```

## References

- [NVIDIA PyTorch for Jetson docs](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [Seeed Studio - Install PyTorch on Jetson](https://wiki.seeedstudio.com/install_torch_on_recomputer/)
- [TensorRT in conda on Jetson](https://nvidia-jetson.piveral.com/jetson-orin-nano/using-tensorrt-in-a-conda-environment-on-jetson-orin-nano/)
- [YolactEdge repo](https://github.com/haotian-liu/yolact_edge)
- [torch2trt repo](https://github.com/NVIDIA-AI-IOT/torch2trt)
