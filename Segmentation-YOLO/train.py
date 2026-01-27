import torch
from ultralytics import YOLO
import argparse

def checkMpsAvailability():
    """
    Check if MPS (Metal Performance Shaders) is available for M-series acceleration
    """
    if torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) is available and will be used for training")
        return 'mps'
    else:
        print("MPS not available, falling back to CPU")
        return 'cpu'

def trainModel(
    modelSize='n',
    epochs=100,
    imgSize=416,
    batchSize=8,
    patience=20,
    name='cone-segmentation',
    resume=False
):
    """
    Train YOLOv8 segmentation model on cone dataset
    """
    device = checkMpsAvailability()

    modelName = f'yolov8{modelSize}-seg.pt'
    print(f"\nLoading {modelName} pretrained model")
    model = YOLO(modelName)

    print(f"\nTraining configuration:")
    print(f"  Model: {modelName}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgSize}")
    print(f"  Batch size: {batchSize}")
    print(f"  Patience: {patience}")
    print(f"  Project name: {name}")

    results = model.train(
        data='dataset.yaml',
        epochs=epochs,
        imgsz=imgSize,
        batch=batchSize,
        device=device,
        patience=patience,
        save=True,
        plots=True,
        project='runs/segment',
        name=name,
        exist_ok=True,
        resume=resume,
        verbose=True,
        workers=4,
        amp=False,
        cache=False
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    metrics = model.val()

    print(f"\nValidation Results:")
    print(f"  Box mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  Box mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Mask mAP@0.5: {metrics.seg.map50:.4f}")
    print(f"  Mask mAP@0.5:0.95: {metrics.seg.map:.4f}")

    bestWeightsPath = f'runs/segment/{name}/weights/best.pt'
    print(f"\nBest weights saved to: {bestWeightsPath}")
    print(f"\nTo run inference:")
    print(f"  python inference.py --weights {bestWeightsPath} --source <image_path>")

    return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 Segmentation on Cone Dataset')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size: n(nano), s(small), m(medium), l(large), x(xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=416,
                       help='Training image size')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--name', type=str, default='cone-segmentation',
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')

    args = parser.parse_args()

    trainModel(
        modelSize=args.model,
        epochs=args.epochs,
        imgSize=args.img_size,
        batchSize=args.batch,
        patience=args.patience,
        name=args.name,
        resume=args.resume
    )

if __name__ == '__main__':
    main()
