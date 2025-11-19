# YolactEdge Setup Guide - Ubuntu (RTX 5070 Ti)

## System Specifications

- **OS:** Ubuntu 24.04.3 LTS (Noble)
- **GPU:** NVIDIA GeForce RTX 5070 Ti (16GB VRAM)
- **Driver:** NVIDIA 580.95.05

## Software Versions

- **Ubuntu:** 24.04.3 LTS (Noble)
- **Python:** 3.10.9
- **PyTorch:** 2.9.1+cu130
- **TensorRT:** 10.14.1.48.post1

## Prerequisites
- Miniconda

## Instructions

## Step 1: Create Python Environment
```bash
conda create -n yolact_edge python=3.10.9 -c conda-forge --override-channels
conda activate yolact_edge
```

## Step 2: Install PyTorch 2.9.1 with CUDA 13.0

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Verify
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
```

## Step 3: Install Dependencies
```bash
pip install cython opencv-python pillow matplotlib GitPython termcolor tensorboard "numpy<2.0" pycocotools
```

## Step 4: Clone YolactEdge
```bash
cd ~
mkdir yolact
cd yolact
git clone https://github.com/haotian-liu/yolact_edge.git
cd yolact_edge
```

## Step 5: Fix setup.py
```bash
nano setup.py
```

Make Changes:
```python
from distutils.extension import Extension
from Cython.Distutils import build_ext
from setuptools import setup, find_packages
import numpy as np # Add this line

cmdclass = {}
cmdclass.update({'build_ext': build_ext})
ext_modules = [
    Extension(
        "cython_nms", 
        ["yolact_edge/utils/cython_nms.pyx"],
        include_dirs=[np.get_include()] # Add this line
    )
]
for e in ext_modules:
    e.cython_directives = {'language_level': "3"}
    
setup(name='yolact_edge',
      version='0.0.1',
      package_dir={'yolact_edge': 'yolact_edge'},
      packages=find_packages(exclude=('data','calib_images','results')) + ['yolact_edge'],
      include_package_data=True,
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      )
```

## Step 6: Build Cython Extensions
```bash
python setup.py build_ext --inplace
```

## Step 7: Install CUDA Toolkit (conda - for compatibility)
```bash
conda install -c conda-forge cudatoolkit=11.8 --no-update-deps
conda install -c conda-forge cudnn=8.9 --no-update-deps

# Verify
conda list | grep cudatoolkit
conda list | grep cudnn
```

**Note:** PyTorch uses CUDA 13.0 internally, but install 11.8 toolkit for compatibility with some tools

## Step 8: Install TensorRT 10.14.1

```bash
pip install tensorrt

# Verify
python -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)"
```

## Step 9: Install torch2trt
```bash
cd ~/yolact
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt

# FIX: TensorRT 10.x removed STRICT_TYPES flag
# Comment out line 666 in torch2trt/torch2trt.py

# Install
python setup.py install

# Verify
python -c "from torch2trt import torch2trt; print('torch2trt works')"
```

## Step 10: Fix CUDA 13.0 Library Paths

```bash
# replace USERNAME with actual username
echo 'export LD_LIBRARY_PATH=/home/USERNAME/miniconda3/envs/yolact_edge/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

echo 'export LD_LIBRARY_PATH=/home/USERNAME/miniconda3/envs/yolact_edge/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc
```

## Step 11: Download Model Weights
```bash
cd ~/yolact/yolact_edge
mkdir weights

# Download from: https://github.com/haotian-liu/yolact_edge/releases
# Get yolact_edge_resnet50_54_800000.pth
# Place in weights/ directory
```

## Commands

### Display Video:
```bash
python eval.py --trained_model=weights/yolact_edge_resnet50_54_800000.pth --video=video.mp4 --display --use_fp16_tensorrt --score_threshold=0.5 --top_k=15
```

### Multi-Frame Processing (Faster):
```bash
python eval.py --trained_model=weights/yolact_edge_resnet50_54_800000.pth --video=video.mp4 --benchmark --use_fp16_tensorrt --video_multiframe=2 --trt_batch_size=2 --score_threshold=0.5 --top_k=15
```

### Process and Save Video:
```bash
python eval.py --trained_model=weights/yolact_edge_resnet50_54_800000.pth --video="input.mp4:output.mp4" --use_fp16_tensorrt --score_threshold=0.5 --top_k=15
```

## Expected FPS
- around 90 fps without tensorrt
- around 220 fps with tensorrt
- around 230 fps with multiframe processing (2 frames)
- around 250 fps with multiframe processing (8 frames)
