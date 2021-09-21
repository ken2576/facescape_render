import os
import glob
import argparse

import imageio
import numpy as np
import matplotlib.pyplot as plt


def read_env(path):
    env = imageio.imread(path)
    return env

def parse_args():
    # Parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str, metavar='PATH',
                        help='path to data root directory')
    parser.add_argument('--out_dir', type=str, metavar='PATH',
                        help='path to the output directory')
    parser.add_argument('--num_rot', type=int,
                        help='number of rotations')
    return parser.parse_args()

def random_rotate(folder, out_folder, num_rot):
    env_paths = sorted(glob.glob(os.path.join(folder, '*.hdr')))
    for env_path in env_paths:
        filename = os.path.split(env_path)[-1][:-4]
        env_map = read_env(env_path)
        print(env_map.shape)
        w = env_map.shape[1]
        rots = np.random.choice([x for x in range(16)], 4, replace=False)

        os.makedirs(out_folder, exist_ok=True)
        for idx, rot in enumerate(rots):
            delta = rot / 16 * w
            deg = rot / 16 * 360
            print(deg)
            delta = delta.astype(int)
            new_env_map = np.roll(env_map, delta, 1)
            imageio.imwrite(os.path.join(out_folder, f'{filename}_{idx:02d}.exr'), new_env_map)
            with open(os.path.join(out_folder, f'{filename}_{idx:02d}.txt'), 'w') as f:
                f.write(f'{deg}')

if __name__ == '__main__':
    imageio.plugins.freeimage.download() # Download needed plugins

    args = parse_args()
    random_rotate(args.root_dir, args.out_dir, args.num_rot)