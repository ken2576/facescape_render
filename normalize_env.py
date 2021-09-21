import os
import glob
import argparse
import re

import numpy as np
import imageio

def normalize_env(env, threshold=0.2, norm_factor=0.4):
    if np.mean(env) < threshold:
        env *= norm_factor / np.mean(env)
    return env

def parse_args():
    # Parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder', type=str, metavar='PATH',
                        help='path to folder with hdr files')
    parser.add_argument('--out_dir', type=str, metavar='PATH',
                        help='path to the output directory')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='path to the output directory')
    parser.add_argument('--norm_factor', type=float, default=0.6,
                        help='path to the output directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    env_maps = sorted(glob.glob(os.path.join(args.folder, '*.hdr')),
        key=lambda f: int(re.sub('\D', '', f)))
    os.makedirs(args.out_dir, exist_ok=True)

    for i in env_maps:
        filename = os.path.split(i)[-1]
        env = imageio.imread(os.path.join(args.folder, i))
        norm_env = normalize_env(env, args.threshold, args.norm_factor)
        imageio.imwrite(os.path.join(args.out_dir, filename), norm_env)