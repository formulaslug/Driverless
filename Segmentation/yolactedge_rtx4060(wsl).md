# YolactEdge Setup Guide - WSL (RTX 4060 Laptop)

## System Specifications

- **OS:** Windows 10/11 with WSL2 (Ubuntu 24.04 LTS)
- **GPU:** NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
- **Driver:** NVIDIA driver 566.26 (installed on Windows, NOT WSL)

## Software Versions

- **Ubuntu:** 24.04.1 LTS (Noble) - WSL2
- **Python:** 3.10.9
- **PyTorch:** 1.13.1+cu117
- **TensorRT:** 8.6.1

## Prerequisites
- WSL2
- Miniconda

## Instructions

## Step 1: Create Python Environment
```bash
conda create -n yolact_edge python=3.10.9 -c conda-forge --override-channels
conda activate yolact_edge
```

## Step 2: Install PyTorch with CUDA 11.7
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Verify
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
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

## Step 7: Install CUDA Toolkit and cuDNN
```bash
conda install -c conda-forge cudatoolkit=11.8 --no-update-deps
conda install -c conda-forge cudnn=8.9.7 --no-update-deps
```

## Step 8: Install TensorRT 8.6.1

### Download TensorRT:
```bash
cd ~
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
```

### Install TensorRT Python:
```bash
cd ~/TensorRT-8.6.1.6/python
pip install tensorrt-8.6.1-cp310-none-linux_x86_64.whl

# Verify
python -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)"
```

### Set Environment Variables:
```bash
echo 'export LD_LIBRARY_PATH=~/TensorRT-8.6.1.6/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 9: Install torch2trt
```bash
cd ~/yolact
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt

python setup.py install

# Verify
python -c "from torch2trt import torch2trt; print('torch2trt works')"
```

## Step 10: Download Model Weights
```bash
cd ~/yolact/yolact_edge
mkdir weights

# Download from: https://github.com/WisconsinAIVision/yolact_edge/tree/master
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
- around 27-29 fps without tensorrt
- around 51 fps with ternsorrt
- around 60 fps with multiframe processing (2 frames)