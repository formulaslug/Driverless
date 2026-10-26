import argparse
import os

from ultralytics import YOLO

defaultWeights = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'runs', 'segment', 'cone-segmentation', 'weights', 'best.pt'
)


# Run on the Jetson: TensorRT engines are specific to the GPU and TensorRT version
# they are built on. The engine is written next to the weights as best.engine,
# where YoloBackend picks it up automatically on CUDA devices.
def exportEngine(weightsPath, imgSize, half):
    model = YOLO(weightsPath)
    return model.export(format='engine', imgsz=imgSize, half=half, device=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export YOLO cone segmentation weights to TensorRT')
    parser.add_argument('--weights', default=defaultWeights)
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--fp32', action='store_true', help='disable FP16')
    args = parser.parse_args()
    print(f"Exported: {exportEngine(args.weights, args.imgsz, not args.fp32)}")
