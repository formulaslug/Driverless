import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch-ssd'))

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config, vgg_ssd_config, cone_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.cone_data_preprocessing import ConeTrainAugmentation
from vision.ssd.ssd import MatchPrior
from vision.utils.misc import Timer, store_labels

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def train(loader, net, criterion, optimizer, device, epoch=-1):
    net.train(True)
    runningLoss = 0.0
    runningRegressionLoss = 0.0
    runningClassificationLoss = 0.0

    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device).float()
        boxes = boxes.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()

        try:
            confidence, locations = net(images)
            regressionLoss, classificationLoss = criterion(confidence, locations, labels, boxes)
            loss = regressionLoss + classificationLoss

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf loss detected at iteration {i}, skipping batch")
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()
        except (RuntimeError, ValueError) as e:
            if "Expected more than 1 value per channel" in str(e):
                logging.warning(f"BatchNorm error at iteration {i} (batch has only 1 sample), skipping batch")
                continue
            else:
                raise

        runningLoss += loss.item()
        runningRegressionLoss += regressionLoss.item()
        runningClassificationLoss += classificationLoss.item()

        if i > 0 and i % 10 == 0:
            avgLoss = runningLoss / 10
            avgRegLoss = runningRegressionLoss / 10
            avgClfLoss = runningClassificationLoss / 10
            logging.info(
                f'Epoch: {epoch}, Step: {i}/{len(loader)}, '
                f'Loss: {avgLoss:.4f}, '
                f'Reg Loss: {avgRegLoss:.4f}, '
                f'Clf Loss: {avgClfLoss:.4f}'
            )

            if avgClfLoss > 1000:
                logging.error(f"Loss explosion detected! Classification loss: {avgClfLoss:.2f}")
                logging.error("Training is unstable. Consider:")
                logging.error("  - Lowering learning rate (try --lr 1e-5)")
                logging.error("  - Increasing batch size (try --batch_size 8)")
                logging.error("  - Using a pretrained model")
                raise RuntimeError("Loss explosion - training stopped")

            runningLoss = 0.0
            runningRegressionLoss = 0.0
            runningClassificationLoss = 0.0

def validate(loader, net, criterion, device):
    net.eval()
    runningLoss = 0.0
    runningRegressionLoss = 0.0
    runningClassificationLoss = 0.0
    num = 0

    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device).float()
        boxes = boxes.to(device).float()
        labels = labels.to(device)

        try:
            with torch.no_grad():
                confidence, locations = net(images)
                regressionLoss, classificationLoss = criterion(confidence, locations, labels, boxes)
                loss = regressionLoss + classificationLoss

            runningLoss += loss.item()
            runningRegressionLoss += regressionLoss.item()
            runningClassificationLoss += classificationLoss.item()
            num += 1
        except (RuntimeError, ValueError) as e:
            if "Expected more than 1 value per channel" in str(e):
                continue
            else:
                raise

    return runningLoss / num, runningRegressionLoss / num, runningClassificationLoss / num

def main():
    parser = argparse.ArgumentParser(description='Train SSD on cone detection dataset')

    parser.add_argument('--data_path', default='./voc_data',
                       help='Path to VOC format dataset')
    parser.add_argument('--net', default='mb2-ssd-lite',
                       choices=['mb1-ssd', 'mb2-ssd-lite', 'vgg16-ssd'],
                       help='Network architecture')
    parser.add_argument('--batch_size', default=4, type=int,
                       help='Batch size for training (minimum 4 recommended to avoid BatchNorm issues)')
    parser.add_argument('--epochs', default=100, type=int,
                       help='Number of training epochs')
    parser.add_argument('--lr', default=1e-4, type=float,
                       help='Learning rate (default: 1e-4, use lower values for stability)')
    parser.add_argument('--momentum', default=0.9, type=float,
                       help='Momentum for SGD')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                       help='Weight decay')
    parser.add_argument('--scheduler', default='cosine',
                       choices=['cosine', 'multi-step'],
                       help='Learning rate scheduler')
    parser.add_argument('--output_dir', default='./checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--resume', default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--base_net_pretrained', default=None,
                       help='Path to pretrained base network weights (e.g., ImageNet MobileNetV2)')
    parser.add_argument('--validation_epochs', default=5, type=int,
                       help='Run validation every N epochs')
    parser.add_argument('--num_workers', default=0, type=int,
                       help='Number of data loading workers (use 0 on macOS)')
    parser.add_argument('--use_cone_config', action='store_true',
                       help='Use cone-optimized config (512x512 input, small anchors for tiny objects)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info('Using CUDA GPU')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info('Using Apple Silicon GPU (MPS)')
    else:
        device = torch.device('cpu')
        logging.info('Using CPU')

    if args.net == 'mb2-ssd-lite':
        createNet = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=1.0, use_cone_config=args.use_cone_config)
        config = cone_ssd_config if args.use_cone_config else mobilenetv1_ssd_config
        if args.use_cone_config:
            logging.info('🎯 Using CONE-OPTIMIZED CONFIG:')
            logging.info(f'   Input size: {config.image_size}x{config.image_size} (vs 300x300 default)')
            logging.info(f'   Anchor sizes: {config.specs[0].box_sizes.min}px - {config.specs[-1].box_sizes.max}px (vs 60-330px default)')
            logging.info(f'   Designed for small objects like traffic cones')
            logging.info('')
            logging.info('🎯 Using CONE-OPTIMIZED AUGMENTATION:')
            logging.info('   ✓ PhotometricDistort (color jittering)')
            logging.info('   ✓ RandomMirror (horizontal flip)')
            logging.info('   ✗ Expand removed (makes tiny cones invisible)')
            logging.info('   ✗ RandomSampleCrop removed (loses tiny cones)')
            logging.info('   → Preserves tiny cone visibility (median: 2.3px × 8.5px)')
    elif args.net == 'mb1-ssd':
        if args.use_cone_config:
            raise ValueError('Cone config only supported for mb2-ssd-lite currently')
        createNet = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'vgg16-ssd':
        if args.use_cone_config:
            raise ValueError('Cone config only supported for mb2-ssd-lite currently')
        createNet = create_vgg_ssd
        config = vgg_ssd_config
    else:
        raise ValueError(f'Unsupported network: {args.net}')

    if args.use_cone_config:
        trainTransform = ConeTrainAugmentation(config.image_size, config.image_mean, config.image_std)
    else:
        trainTransform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)

    targetTransform = MatchPrior(config.priors, config.center_variance,
                                config.size_variance, 0.5)
    testTransform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info(f'Loading training dataset from {args.data_path}')
    trainDataset = VOCDataset(args.data_path, transform=trainTransform,
                             target_transform=targetTransform)

    numClasses = len(trainDataset.class_names)
    logging.info(f'Number of classes: {numClasses}')
    logging.info(f'Classes: {trainDataset.class_names}')
    logging.info(f'Training images: {len(trainDataset)}')

    trainLoader = DataLoader(trainDataset, args.batch_size,
                           num_workers=args.num_workers,
                           shuffle=True)

    valDataset = VOCDataset(args.data_path, transform=testTransform,
                           target_transform=targetTransform, is_test=True)
    logging.info(f'Validation images: {len(valDataset)}')

    valLoader = DataLoader(valDataset, args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=False)

    logging.info('Creating network...')
    net = createNet(numClasses)

    if args.base_net_pretrained and not args.resume:
        logging.info(f'Loading pretrained base network from: {args.base_net_pretrained}')
        try:
            pretrainedDict = torch.load(args.base_net_pretrained, map_location='cpu')

            modelDict = net.state_dict()
            pretrainedDictFiltered = {}

            logging.info(f'DEBUG: Pretrained dict has {len(pretrainedDict)} keys')
            logging.info(f'DEBUG: Model dict has {len(modelDict)} keys')

            pretrainedKeys = list(pretrainedDict.keys())[:5]
            modelKeys = [k for k in modelDict.keys() if k.startswith('base_net.')][:5]
            logging.info(f'DEBUG: Sample pretrained keys: {pretrainedKeys}')
            logging.info(f'DEBUG: Sample model base_net keys: {modelKeys}')

            matchedCount = 0
            shapeMismatchCount = 0
            keyNotFoundCount = 0

            for k, v in pretrainedDict.items():
                newKey = k
                if not k.startswith('base_net.'):
                    newKey = 'base_net.' + k

                if newKey in modelDict:
                    if modelDict[newKey].shape == v.shape:
                        pretrainedDictFiltered[newKey] = v
                        matchedCount += 1
                    else:
                        shapeMismatchCount += 1
                        if shapeMismatchCount <= 3:
                            logging.warning(f'  Shape mismatch: {newKey} - pretrained {v.shape} vs model {modelDict[newKey].shape}')
                else:
                    keyNotFoundCount += 1
                    if keyNotFoundCount <= 3:
                        logging.warning(f'  Key not found in model: {newKey}')

            logging.info(f'DEBUG: Matched: {matchedCount}, Shape mismatches: {shapeMismatchCount}, Keys not found: {keyNotFoundCount}')

            modelDict.update(pretrainedDictFiltered)
            net.load_state_dict(modelDict)

            logging.info(f'✓ Loaded {len(pretrainedDictFiltered)} pretrained layers')
            logging.info('  Base network initialized with ImageNet weights')
            logging.info('  Detection heads will be trained from scratch')
        except Exception as e:
            logging.warning(f'Could not load pretrained weights: {e}')
            logging.warning('Training from scratch instead')
            import traceback
            traceback.print_exc()

    if args.resume:
        logging.info(f'Resuming from checkpoint: {args.resume}')
        net.load(args.resume)

    net = net.to(device)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                           center_variance=0.1, size_variance=0.2, device=device)

    if args.base_net_pretrained and not args.resume:
        logging.info('Using different learning rates: base_net (pretrained) vs heads (from scratch)')
        baseNetParams = []
        headParams = []

        for name, param in net.named_parameters():
            if 'base_net' in name:
                baseNetParams.append(param)
            else:
                headParams.append(param)

        optimizer = torch.optim.SGD([
            {'params': baseNetParams, 'lr': args.lr * 0.1},
            {'params': headParams, 'lr': args.lr}
        ], momentum=args.momentum, weight_decay=args.weight_decay)

        logging.info(f'  Base network LR: {args.lr * 0.1} (fine-tuning)')
        logging.info(f'  Detection heads LR: {args.lr} (training from scratch)')
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        logging.info('Using Cosine Annealing LR scheduler')
    else:
        milestones = [int(args.epochs * 0.6), int(args.epochs * 0.8)]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        logging.info(f'Using MultiStep LR scheduler with milestones: {milestones}')

    logging.info(f'Starting training for {args.epochs} epochs...')
    logging.info(f'Learning rate: {args.lr}, Batch size: {args.batch_size}')

    for epoch in range(args.epochs):
        train(trainLoader, net, criterion, optimizer, device, epoch)
        scheduler.step()

        if epoch % args.validation_epochs == 0 or epoch == args.epochs - 1:
            valLoss, valRegLoss, valClfLoss = validate(valLoader, net, criterion, device)
            logging.info(
                f'Epoch {epoch} Validation - '
                f'Loss: {valLoss:.4f}, '
                f'Reg Loss: {valRegLoss:.4f}, '
                f'Clf Loss: {valClfLoss:.4f}'
            )

            modelPath = os.path.join(args.output_dir,
                                    f'{args.net}-epoch-{epoch}-loss-{valLoss:.4f}.pth')
            net.save(modelPath)
            logging.info(f'Saved checkpoint: {modelPath}')

    finalPath = os.path.join(args.output_dir, f'{args.net}-final.pth')
    net.save(finalPath)

    labelPath = os.path.join(args.output_dir, 'cone-labels.txt')
    store_labels(labelPath, trainDataset.class_names)

    logging.info(f'Training complete! Final model saved to: {finalPath}')
    logging.info(f'Labels saved to: {labelPath}')

if __name__ == '__main__':
    main()
