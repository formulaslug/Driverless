import argparse
import os
import sys
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch-ssd'))

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor

def loadModel(modelPath, labelPath, netType='mb2-ssd-lite', useConeConfig=False):
    """
    Load trained SSD model

    Args:
        modelPath: Path to model checkpoint
        labelPath: Path to labels file
        netType: Network type (mb2-ssd-lite, mb1-ssd, vgg16-ssd)
        useConeConfig: Whether model was trained with cone config (512x512, small anchors)
    """
    with open(labelPath, 'r') as f:
        classNames = [name.strip() for name in f.readlines()]

    numClasses = len(classNames)

    if netType == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(numClasses, is_test=True, use_cone_config=useConeConfig)
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    elif netType == 'mb1-ssd':
        if useConeConfig:
            raise ValueError('Cone config only supported for mb2-ssd-lite')
        net = create_mobilenetv1_ssd(numClasses, is_test=True)
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    elif netType == 'vgg16-ssd':
        if useConeConfig:
            raise ValueError('Cone config only supported for mb2-ssd-lite')
        net = create_vgg_ssd(numClasses, is_test=True)
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    else:
        raise ValueError(f'Unsupported network type: {netType}')

    net.load(modelPath)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    net = net.to(device)

    return predictor, classNames, device

def visualizeDetections(image, boxes, labels, probs, classNames, threshold=0.5):
    """
    Draw bounding boxes on image
    """
    colors = {
        'seg_orange_cone': (255, 128, 0),
        'seg_yellow_cone': (255, 255, 0),
        'seg_large_orange_cone': (255, 69, 0),
        'seg_blue_cone': (0, 0, 255),
        'seg_unknown_cone': (128, 128, 128)
    }

    for i in range(boxes.size(0)):
        if probs[i] < threshold:
            continue

        box = boxes[i, :]
        label = classNames[labels[i]]
        prob = probs[i]

        color = colors.get(label, (0, 255, 0))

        cv2.rectangle(image,
                     (int(box[0]), int(box[1])),
                     (int(box[2]), int(box[3])),
                     color, 2)

        text = f'{label}: {prob:.2f}'
        (textWidth, textHeight), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(image,
                     (int(box[0]), int(box[1]) - textHeight - baseline),
                     (int(box[0]) + textWidth, int(box[1])),
                     color, -1)

        cv2.putText(image, text,
                   (int(box[0]), int(box[1]) - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def runInference(predictor, imagePath, classNames, threshold=0.5):
    """
    Run inference on a single image
    """
    origImage = cv2.imread(imagePath)
    if origImage is None:
        raise ValueError(f'Could not read image: {imagePath}')

    image = cv2.cvtColor(origImage, cv2.COLOR_BGR2RGB)

    boxes, labels, probs = predictor.predict(image, 10, threshold)

    return origImage, boxes, labels, probs

def main():
    parser = argparse.ArgumentParser(description='SSD Inference on cone images')

    parser.add_argument('--model', required=True,
                       help='Path to trained model')
    parser.add_argument('--labels', required=True,
                       help='Path to labels file')
    parser.add_argument('--image', required=True,
                       help='Path to input image')
    parser.add_argument('--output', default=None,
                       help='Path to save output image')
    parser.add_argument('--net', default='mb2-ssd-lite',
                       choices=['mb1-ssd', 'mb2-ssd-lite', 'vgg16-ssd'],
                       help='Network architecture')
    parser.add_argument('--threshold', default=0.5, type=float,
                       help='Detection confidence threshold')
    parser.add_argument('--use_cone_config', action='store_true',
                       help='Use cone-optimized config (must match training config)')

    args = parser.parse_args()

    print('Loading model...')
    predictor, classNames, device = loadModel(args.model, args.labels, args.net, args.use_cone_config)
    print(f'Model loaded. Using device: {device}')
    print(f'Classes: {classNames}')

    print(f'Running inference on {args.image}...')
    origImage, boxes, labels, probs = runInference(predictor, args.image,
                                                   classNames, args.threshold)

    print(f'Found {len(probs)} detections:')
    for i in range(len(probs)):
        if probs[i] >= args.threshold:
            label = classNames[labels[i]]
            box = boxes[i, :]
            print(f'  {label}: {probs[i]:.3f} at [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]')

    resultImage = visualizeDetections(origImage, boxes, labels, probs,
                                     classNames, args.threshold)

    if args.output:
        outputPath = args.output
    else:
        base, ext = os.path.splitext(args.image)
        outputPath = f'{base}_detection{ext}'

    cv2.imwrite(outputPath, resultImage)
    print(f'Saved result to: {outputPath}')

if __name__ == '__main__':
    main()
