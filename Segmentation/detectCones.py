import sys
import os
import argparse

sys.path.insert(0, 'Yolact_minimal')

def main():
    parser = argparse.ArgumentParser(description='Detect cones using trained YOLACT model')
    parser.add_argument('--weight', type=str, required=True,
                       help='Path to trained weights (e.g., Yolact_minimal/weights/best_32.5_res50_cone_40000.pth)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image or folder of images')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file')
    parser.add_argument('--img_size', type=int, default=544,
                       help='Image size for inference')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Confidence threshold for detections (0-1)')
    parser.add_argument('--hide_mask', action='store_true',
                       help='Hide segmentation masks (show only boxes)')
    parser.add_argument('--hide_bbox', action='store_true',
                       help='Hide bounding boxes (show only masks)')
    parser.add_argument('--hide_score', action='store_true',
                       help='Hide confidence scores')
    parser.add_argument('--real_time', action='store_true',
                       help='Display video detection in real-time (no saving)')

    args = parser.parse_args()

    if not os.path.exists(args.weight):
        print(f"Error: Weight file not found: {args.weight}")
        print("\nAvailable weights in Yolact_minimal/weights/:")
        if os.path.exists('Yolact_minimal/weights'):
            weights = [f for f in os.listdir('Yolact_minimal/weights') if f.endswith('.pth')]
            if weights:
                for w in weights:
                    print(f"  - {w}")
            else:
                print("  (No .pth files found)")
        return

    if args.image is None and args.video is None:
        print("Error: Must specify either --image or --video")
        return

    print("="*60)
    print("YOLACT Cone Detection")
    print("="*60)
    print(f"Weight: {args.weight}")
    print(f"Image size: {args.img_size}")
    print(f"Threshold: {args.threshold}")
    if args.image:
        print(f"Input: {args.image}")
        print(f"Output: Yolact_minimal/results/images/")
    elif args.video:
        print(f"Input: {args.video}")
        if args.real_time:
            print(f"Output: Real-time display")
        else:
            print(f"Output: Yolact_minimal/results/videos/")
    print("="*60)
    print()

    os.chdir('Yolact_minimal')

    cmd_parts = [
        'python', 'detect.py',
        '--weight', f'../{args.weight}' if not args.weight.startswith('Yolact_minimal') else args.weight.replace('Yolact_minimal/', ''),
        '--img_size', str(args.img_size),
        '--visual_thre', str(args.threshold)
    ]

    if args.image:
        img_path = f'../{args.image}' if not args.image.startswith('Yolact_minimal') else args.image.replace('Yolact_minimal/', '')
        cmd_parts.extend(['--image', img_path])
    elif args.video:
        vid_path = f'../{args.video}' if not args.video.startswith('Yolact_minimal') else args.video.replace('Yolact_minimal/', '')
        cmd_parts.extend(['--video', vid_path])

    if args.hide_mask:
        cmd_parts.append('--hide_mask')
    if args.hide_bbox:
        cmd_parts.append('--hide_bbox')
    if args.hide_score:
        cmd_parts.append('--hide_score')
    if args.real_time:
        cmd_parts.append('--real_time')

    cmd = ' '.join(cmd_parts)
    print(f"Running: {cmd}\n")

    os.system(cmd)

if __name__ == '__main__':
    main()
