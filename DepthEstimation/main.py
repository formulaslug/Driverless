import os
import sys
import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms
import matplotlib.cm as cm
import matplotlib.colors as colors
from PIL import Image

sys.path.append('./Lite-Mono')
import networks
from layers import disp_to_depth

MODEL_CONFIGS = {
    'tiny': {
        'name': 'lite-mono-tiny',
        'folder': 'lite-mono-tiny_640x192',
        'description': 'Lightest/fastest (2.2M params, 640x192)'
    },
    'best': {
        'name': 'lite-mono-8m',
        'folder': 'lite-mono-8m_1024x320',
        'description': 'Best quality/heaviest (8.7M params, 1024x320)'
    }
}


class DepthEstimator:
    def __init__(self, weightsFolder, model="lite-mono-tiny", useCuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and useCuda else "cpu")

        encoderPath = os.path.join(weightsFolder, "encoder.pth")
        decoderPath = os.path.join(weightsFolder, "depth.pth")

        encoderDict = torch.load(encoderPath, map_location=self.device)
        decoderDict = torch.load(decoderPath, map_location=self.device)

        self.feedHeight = encoderDict['height']
        self.feedWidth = encoderDict['width']

        print(f"Loading {model} model ({self.feedWidth}x{self.feedHeight}) on {self.device}")

        self.encoder = networks.LiteMono(model=model, height=self.feedHeight, width=self.feedWidth)
        modelDict = self.encoder.state_dict()
        self.encoder.load_state_dict({k: v for k, v in encoderDict.items() if k in modelDict})
        self.encoder.to(self.device)
        self.encoder.eval()

        self.depthDecoder = networks.DepthDecoder(self.encoder.num_ch_enc, scales=range(3))
        depthModelDict = self.depthDecoder.state_dict()
        self.depthDecoder.load_state_dict({k: v for k, v in decoderDict.items() if k in depthModelDict})
        self.depthDecoder.to(self.device)
        self.depthDecoder.eval()

        self.toTensor = transforms.ToTensor()

    def processFrame(self, frame):
        originalHeight, originalWidth = frame.shape[:2]

        frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        framePil = Image.fromarray(frameRgb)
        frameResized = framePil.resize((self.feedWidth, self.feedHeight), Image.LANCZOS)

        inputTensor = self.toTensor(frameResized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.encoder(inputTensor)
            outputs = self.depthDecoder(features)

        disp = outputs[("disp", 0)]
        dispResized = torch.nn.functional.interpolate(
            disp, (originalHeight, originalWidth), mode="bilinear", align_corners=False)

        dispNp = dispResized.squeeze().cpu().numpy()

        vmax = np.percentile(dispNp, 95)
        normalizer = colors.Normalize(vmin=dispNp.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped = (mapper.to_rgba(dispNp)[:, :, :3] * 255).astype(np.uint8)

        return colormapped, dispNp

    def processVideo(self, videoPath, outputFolder="output", saveFrames=True, displayLive=False):
        os.makedirs(outputFolder, exist_ok=True)

        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {videoPath}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Processing video: {frameCount} frames at {fps:.2f} FPS ({width}x{height})")

        outputVideoPath = os.path.join(outputFolder, "depth_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outVideo = cv2.VideoWriter(outputVideoPath, fourcc, fps, (width, height))

        frameIdx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            depthColor, depthRaw = self.processFrame(frame)
            depthBgr = cv2.cvtColor(depthColor, cv2.COLOR_RGB2BGR)

            outVideo.write(depthBgr)

            if saveFrames and frameIdx % 30 == 0:
                cv2.imwrite(os.path.join(outputFolder, f"frame_{frameIdx:06d}_depth.png"), depthBgr)
                np.save(os.path.join(outputFolder, f"frame_{frameIdx:06d}_depth.npy"), depthRaw)

            if displayLive:
                combined = np.hstack((frame, depthBgr))
                cv2.imshow('Original | Depth', combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frameIdx += 1
            if frameIdx % 30 == 0:
                print(f"Processed {frameIdx}/{frameCount} frames ({100*frameIdx/frameCount:.1f}%)")

        cap.release()
        outVideo.release()
        if displayLive:
            cv2.destroyAllWindows()

        print(f"\nComplete! Processed {frameIdx} frames")
        print(f"Output video: {outputVideoPath}")
        return outputVideoPath


def main():
    parser = argparse.ArgumentParser(description='Lite-Mono Depth Estimation')
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'best'],
                        help='Model to use: tiny (fastest) or best (highest quality)')
    parser.add_argument('--video', type=str, default='./TestData/test1.mp4',
                        help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                        help='Output folder (default: ./output_<model>)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA if available')

    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    weightsFolder = f"./weights/{config['folder']}"
    outputFolder = args.output or f"./output_{args.model}"

    print(f"Using model: {config['description']}")

    if not os.path.exists(weightsFolder):
        print(f"Error: Weights folder not found: {weightsFolder}")
        print("Please download the model weights first")
        return

    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        return

    estimator = DepthEstimator(weightsFolder, model=config['name'], useCuda=args.cuda)
    estimator.processVideo(args.video, outputFolder=outputFolder, saveFrames=True, displayLive=False)


if __name__ == '__main__':
    main()
