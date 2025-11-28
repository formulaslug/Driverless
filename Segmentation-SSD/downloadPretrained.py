#!/usr/bin/env python
"""
Download pretrained MobileNetV2 weights from torchvision
and convert to format compatible with pytorch-ssd
"""
import os
import torch
import torchvision

def downloadMobileNetV2Pretrained(outputDir='./models'):
    """
    Download MobileNetV2 pretrained on ImageNet from torchvision
    """
    os.makedirs(outputDir, exist_ok=True)

    print("Downloading MobileNetV2 pretrained weights from torchvision...")

    model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')

    stateDictPath = os.path.join(outputDir, 'mobilenet_v2_imagenet.pth')

    torch.save(model.state_dict(), stateDictPath)

    print(f"✓ Saved pretrained MobileNetV2 weights to: {stateDictPath}")
    print(f"  Model trained on ImageNet with 1000 classes")
    print(f"  File size: {os.path.getsize(stateDictPath) / 1024 / 1024:.2f} MB")

    featureStateDictPath = os.path.join(outputDir, 'mobilenet_v2_imagenet_features.pth')
    featureStateDict = {}
    for key, value in model.state_dict().items():
        if key.startswith('features.'):
            featureStateDict[key] = value

    torch.save(featureStateDict, featureStateDictPath)
    print(f"✓ Saved feature extractor only to: {featureStateDictPath}")
    print(f"  (For base_net initialization)")

    return stateDictPath, featureStateDictPath

if __name__ == '__main__':
    downloadMobileNetV2Pretrained()
    print("\nPretrained weights ready!")
    print("Use with: python train.py --base_net_pretrained ./models/mobilenet_v2_imagenet_features.pth ...")
