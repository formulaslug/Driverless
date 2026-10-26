import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline', 'nodes'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline', 'backends'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline', 'motion'))

import pipelinePaths  # noqa: F401  adds sibling perception dirs to sys.path
from graph import Graph
from cameraModel import CameraModel
from videoSource import VideoSource, CameraSource
from captureNode import CaptureNode
from segmentationNode import SegmentationNode
from distanceNode import DistanceNode
from motionNode import MotionNode
from localizationNode import LocalizationNode
from planningNode import PlanningNode
from videoWriterNode import VideoWriterNode
from segmentationBackends import createSegmentationBackend
from motionEstimator import createMotionEstimator


def openSource(args):
    if args.camera is not None:
        return CameraSource(args.camera, targetFps=args.targetFps, maxFrames=args.maxFrames)
    return VideoSource(args.video, targetFps=args.targetFps, maxFrames=args.maxFrames, startFrame=args.startFrame)


def buildPipeline(args):
    source = openSource(args)
    camera = CameraModel.fromConfig(source.width, source.height, configPath=args.cameraConfig, fov=args.fov)

    segBackend = createSegmentationBackend(args.segBackend, device=args.device)
    motionEstimator = createMotionEstimator(args.motion)

    graph = Graph()
    capture = graph.add(CaptureNode(source, camera))
    segmentation = graph.add(SegmentationNode(segBackend))
    distance = graph.add(DistanceNode(camera))
    motion = graph.add(MotionNode(motionEstimator))
    localization = graph.add(LocalizationNode())
    planning = graph.add(PlanningNode())
    videoWriter = graph.add(VideoWriterNode(args.output, fps=source.outputFps))

    capture.subscribe(segmentation)
    capture.subscribe(motion)
    capture.subscribe(videoWriter)

    segmentation.subscribe(distance)
    segmentation.subscribe(localization)
    segmentation.subscribe(videoWriter)

    distance.subscribe(localization)
    distance.subscribe(videoWriter)
    motion.subscribe(localization)

    localization.subscribe(planning)
    localization.subscribe(videoWriter)
    planning.subscribe(videoWriter)

    return graph


def parseArgs():
    parser = argparse.ArgumentParser(description='Video or camera -> path-planning pipeline (concurrent node graph)')
    parser.add_argument('--video', default='SampleData/driverless.mp4')
    parser.add_argument('--camera', default=None, help='live camera: device index or GStreamer pipeline (overrides --video)')
    parser.add_argument('--output', default='output/pipeline_annotated.mp4')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--target-fps', dest='targetFps', type=float, default=10.0)
    parser.add_argument('--max-frames', dest='maxFrames', type=int, default=None)
    parser.add_argument('--start-frame', dest='startFrame', type=int, default=0)
    parser.add_argument('--camera-config', dest='cameraConfig', default=None)
    parser.add_argument('--fov', type=float, default=None, help='horizontal FOV override in degrees (default: camera.json)')
    parser.add_argument('--motion', default='none', choices=['imu', 'none'])
    parser.add_argument('--seg-backend', dest='segBackend', default='yolo', choices=['yolo', 'yolactedge'])
    return parser.parse_args()


def main():
    args = parseArgs()
    print(f"Building pipeline: seg={args.segBackend} motion={args.motion}")
    graph = buildPipeline(args)
    print(f"Processing {args.camera if args.camera is not None else args.video} -> {args.output}")
    graph.run()
    print("Done.")


if __name__ == '__main__':
    main()
