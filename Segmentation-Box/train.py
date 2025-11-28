import argparse
import collections
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch-retinanet'))

from coneDataset import ConeDataset
from retinanet import model
from retinanet.dataloader import collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer

def main(args=None):
    parser = argparse.ArgumentParser(description='Training script for RetinaNet on cone dataset.')

    parser.add_argument('--data_path', help='Path to dataset root directory', required=True)
    parser.add_argument('--coco_json', help='Path to COCO format JSON file', required=True)
    parser.add_argument('--depth', help='ResNet depth (18, 34, 50, 101, 152)', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=4)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-6)
    parser.add_argument('--output_dir', help='Output directory for checkpoints', default='./checkpoints')
    parser.add_argument('--resume', help='Path to checkpoint to resume from', default=None)

    args = parser.parse_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    datasetTrain = ConeDataset(
        rootDir=args.data_path,
        cocoJsonPath=args.coco_json,
        transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()])
    )

    numClasses = datasetTrain.numClasses()
    print(f'Number of classes: {numClasses}')
    print(f'Classes: {datasetTrain.classes}')
    print(f'Number of training images: {len(datasetTrain)}')

    sampler = AspectRatioBasedSampler(datasetTrain, batch_size=args.batch_size, drop_last=False)
    dataloaderTrain = DataLoader(datasetTrain, num_workers=2, collate_fn=collater, batch_sampler=sampler)

    if args.depth == 18:
        retinanet = model.resnet18(num_classes=numClasses, pretrained=True)
    elif args.depth == 34:
        retinanet = model.resnet34(num_classes=numClasses, pretrained=True)
    elif args.depth == 50:
        retinanet = model.resnet50(num_classes=numClasses, pretrained=True)
    elif args.depth == 101:
        retinanet = model.resnet101(num_classes=numClasses, pretrained=True)
    elif args.depth == 152:
        retinanet = model.resnet152(num_classes=numClasses, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA GPU')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using Apple Silicon GPU (MPS)')
    else:
        print('Using CPU')

    retinanet = retinanet.to(device)

    if args.resume:
        print(f'Loading checkpoint from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            retinanet.load_state_dict(checkpoint['model_state_dict'])
            startEpoch = checkpoint.get('epoch', 0) + 1
        else:
            retinanet = checkpoint
            startEpoch = 0
    else:
        startEpoch = 0

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    print(f'Learning rate: {args.lr}')

    lossHist = collections.deque(maxlen=500)

    print('Starting training...')

    for epochNum in range(startEpoch, args.epochs):
        retinanet.train()
        retinanet.freeze_bn()

        epochLoss = []

        for iterNum, data in enumerate(dataloaderTrain):
            try:
                optimizer.zero_grad()

                imgs = data['img'].to(device).float()
                annots = data['annot'].float().to(device)

                classificationLoss, regressionLoss = retinanet([imgs, annots])

                classificationLoss = classificationLoss.mean()
                regressionLoss = regressionLoss.mean()

                loss = classificationLoss + regressionLoss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                lossHist.append(float(loss))
                epochLoss.append(float(loss))

                if iterNum % 10 == 0:
                    print(
                        f'Epoch: {epochNum} | Iteration: {iterNum} | '
                        f'Classification loss: {float(classificationLoss):.5f} | '
                        f'Regression loss: {float(regressionLoss):.5f} | '
                        f'Running loss: {np.mean(lossHist):.5f}'
                    )

                del classificationLoss
                del regressionLoss
            except Exception as e:
                print(f'Error in iteration {iterNum}: {e}')
                continue

        scheduler.step(np.mean(epochLoss))

        checkpointPath = os.path.join(args.output_dir, f'cone_retinanet_epoch_{epochNum}.pt')
        torch.save({
            'epoch': epochNum,
            'model_state_dict': retinanet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(epochLoss),
        }, checkpointPath)
        print(f'Saved checkpoint to {checkpointPath}')

    retinanet.eval()

    finalPath = os.path.join(args.output_dir, 'cone_retinanet_final.pt')
    torch.save({
        'model_state_dict': retinanet.state_dict(),
        'classes': datasetTrain.classes,
        'num_classes': numClasses
    }, finalPath)
    print(f'Training complete! Final model saved to {finalPath}')

if __name__ == '__main__':
    main()
