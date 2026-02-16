from yolact_edge.data import cfg, set_cfg
from yolact_edge.yolact import Yolact
from yolact_edge.utils.augmentations import BaseTransform, FastBaseTransform
from yolact_edge.utils.functions import SavePath
from yolact_edge.layers.output_utils import postprocess
from yolact_edge.utils.tensorrt import convert_to_tensorrt

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse

import cv2
import logging



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT Live Inference')
    parser.add_argument('--trained_model',
                        default=None, type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--config', default=None,
                         help='The config object to use.')
    parser.add_argument('--deterministic', default=False, dest='deterministic', action='store_true',
                        help='Whether to enable deterministic flags of PyTorch for deterministic results.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--yolact_transfer', dest='yolact_transfer', action='store_true',
                        help='Split pretrained FPN weights to two phase FPN (for models trained by YOLACT).')
    parser.add_argument('--coco_transfer', dest='coco_transfer', action='store_true',
                        help='[Deprecated] Split pretrained FPN weights to two phase FPN (for models trained by YOLACT).')
    parser.add_argument('--drop_weights', default=None, type=str,
                        help='Drop specified weights (split by comma) from existing model.')
    parser.add_argument('--calib_images', default=None, type=str,
                        help='Directory of images for TensorRT INT8 calibration, for explanation of this field, please refer to `calib_images` in `data/config.py`.')
    parser.add_argument('--trt_batch_size', default=1, type=int,
                        help='Maximum batch size to use during TRT conversion. This has to be greater than or equal to the batch size the model will take during inferece.')
    parser.add_argument('--disable_tensorrt', default=False, dest='disable_tensorrt', action='store_true',
                        help='Don\'t use TensorRT optimization when specified.')
    parser.add_argument('--use_fp16_tensorrt', default=False, dest='use_fp16_tensorrt', action='store_true',
                        help='This replaces all TensorRT INT8 optimization with FP16 optimization when specified.')
    parser.add_argument('--use_tensorrt_safe_mode', default=False, dest='use_tensorrt_safe_mode', action='store_true',
                        help='This enables the safe mode that is a workaround for various TensorRT engine issues.')

    parser.set_defaults(crop=True)

    global args
    args = parser.parse_args(argv)

def live_camera(net:Yolact, camera_idx:int=0):

    #open webcam
    vid = cv2.VideoCapture(camera_idx)
    if not vid.isOpened():
        print(f'Could not open camera')
        return

    #get resolution needed for postprocess() and masks.view(-1, frame_height, frame_width)
    frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    transform = FastBaseTransform() #resize and normalize frames

    frame_idx = 0
    every_k_frames = 5      #full backbone runs every 5 frames
    moving_statistics = {"conf_hist": []}

    try:
        net.detect.use_fast_nms = True      #removes duplicate detections for same object when network runs
        cfg.mask_proto_debug = False
        while True:
            frame = torch.from_numpy(vid.read()[1]).cuda().float()
            batch = transform(frame.unsqueeze(0))

            if frame_idx % every_k_frames == 0 or cfg.flow.warp_mode == 'none':  #every kth frame or if flow is disabled, run full backbone
                extras = {"backbone": "full", "interrupt": False, "keep_statistics": True,
                        "moving_statistics": moving_statistics}
                with torch.no_grad():
                    net_outs = net(batch, extras=extras)
                moving_statistics["feats"] = net_outs["feats"]      #saves features so the next 4 frames can reuse
                moving_statistics["lateral"] = net_outs["lateral"]  #FPN's lateral connections
            else:
                extras = {"backbone": "partial", "interrupt": False, "keep_statistics": False,
                        "moving_statistics": moving_statistics}
                with torch.no_grad():
                    net_outs = net(batch, extras=extras)

            preds = net_outs["pred_outs"]

            classes, scores, boxes, masks = postprocess(preds, frame_width, frame_height, crop_masks=args.crop, score_threshold=args.score_threshold)

            if classes.size(0) > 0:
                n = min(args.top_k, classes.size(0))
                det_classes = classes[:n].cpu().numpy()
                det_scores  = scores[:n].cpu().numpy()
                det_boxes   = boxes[:n].cpu().numpy()
                det_masks   = masks[:n].view(-1, frame_height, frame_width).cpu().numpy()
                det_names   = [cfg.dataset.class_names[c] for c in det_classes]

                frame_detections = {
                    "classes": det_classes,
                    "names": det_names,
                    "scores": det_scores,
                    "boxes": det_boxes,
                    "masks": det_masks,
                }

                print(frame_detections)

            frame_idx += 1

    except KeyboardInterrupt:
        print('Stopping early.')

    vid.release()
    print()

if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    from yolact_edge.utils.logging_helper import setup_logger
    setup_logger(logging_level=logging.INFO)
    logger = logging.getLogger("yolact.eval")

    with torch.no_grad():
        if args.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            if args.deterministic:
                cudnn.deterministic = True
                cudnn.benchmark = False
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        logger.info('Loading model...')
        net = Yolact(training=False)
        if args.trained_model is not None:
            net.load_weights(args.trained_model, args=args)
        else:
            logger.warning("No weights loaded!")
        net.eval()
        logger.info('Model loaded.')

        convert_to_tensorrt(net, cfg, args, transform=BaseTransform())

        if args.cuda:
            net = net.cuda()

        live_camera(net, camera_idx=int(args.video))
