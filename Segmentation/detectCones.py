#!/usr/bin/env python3
import sys
import os
import subprocess

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolact_dir = os.path.join(script_dir, 'yolact_edge')

    args = sys.argv[1:]

    default_args = [
        '--config=yolact_edge_mobilenetv2_cone_config',
        '--score_threshold=0.3',
        '--top_k=100',
    ]

    provided_flags = {arg.split('=')[0].lstrip('-') for arg in args if '=' in arg or arg.startswith('--')}

    final_args = []
    for default_arg in default_args:
        flag = default_arg.split('=')[0].lstrip('-')
        if flag not in provided_flags and not any(arg.startswith('--' + flag) for arg in args):
            final_args.append(default_arg)

    final_args.extend(args)

    cmd = [sys.executable, 'eval.py'] + final_args

    print(f"Running inference in {yolact_dir}")
    print(f"Command: {' '.join(cmd)}\n")

    os.chdir(yolact_dir)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
