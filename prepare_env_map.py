import os
import glob
import json
import argparse

import torch
import imageio
import numpy as np

from resample import get_3x4_RT_matrix_from_blender, pano_inverse_warp, downsample

def get_env_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        data = json.load(f)
    env_path = data[0]['env_path']
    return env_path

def read_env(path):
    env = imageio.imread(path)
    return env

def parse_args():
    # Parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str, metavar='PATH',
                        help='path to data root directory')
    parser.add_argument('--read_txt', action='store_true',
                        help='use txt file to find env map')
    parser.add_argument('--tgt_wh', type=int, nargs='+',
                        default=[32, 16],
                        help='path to data folder')
    return parser.parse_args()

def proc_folder(args, folder):
    out_folder = os.path.join(folder, 'envmap')
    os.makedirs(out_folder, exist_ok=True)

    new_hw = (args.tgt_wh[1], args.tgt_wh[0])

    device = 'cuda'
    print(folder)

    
    if args.read_txt:
        txt = os.path.join(folder, 'commandline_args.txt')
        # env_paths = ['F:\\facescape\\env_map\\lilienstein_1k.hdr'] * 30
        env_paths = [get_env_from_txt(txt)] * 30
    else:
        env_paths = np.load(os.path.join(folder, 'env_maps.npy'))

    exts = np.load(os.path.join(folder, 'cam_exts.npy'))

    # Source rotation (hard-coded)
    src_rot = np.array([
        [1., 0, 0],
        [0, 0, -1.],
        [0, 1., 0]
    ])

    # A 90-degree rotation as the env map is rotated by default
    deg = -90.0
    rad = np.deg2rad(deg)
    fix1 = torch.FloatTensor([
        [np.cos(rad), 0, np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])[None, ...]

    # Second rotation to bring back to the correct orientation (seam behind the person)
    deg = 180.0
    rad = np.deg2rad(deg)
    fix2 = torch.FloatTensor([
        [np.cos(rad), 0, np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])[None, ...]

    for idx, (ext, env_path) in enumerate(zip(exts, env_paths)):
        env_map = read_env(env_path)
        env_map = torch.FloatTensor(env_map).permute([2, 0, 1])[None, ...]
        tgt_rot = get_3x4_RT_matrix_from_blender(ext)[:3, :3]
        trnsf = fix1 @ torch.FloatTensor(src_rot @ np.linalg.inv(tgt_rot)) @ fix2

        # Warp according to camera orientation
        warped = pano_inverse_warp(env_map.to(device), trnsf.to(device))
        resized = downsample(warped, new_hw).squeeze().permute([1, 2, 0]).cpu().numpy()
        resized = resized[:, ::-1] # fix for irradiance
        imageio.imwrite(os.path.join(out_folder, f'{idx:03d}_lightcam.hdr'), resized)

        # Global env map
        trnsf = fix1 @ fix2
        warped = pano_inverse_warp(env_map.to(device), trnsf.to(device))
        resized = downsample(warped, new_hw).squeeze().permute([1, 2, 0]).cpu().numpy()
        resized = resized[:, ::-1] # fix for irradiance
        imageio.imwrite(os.path.join(out_folder, f'{idx:03d}_light.hdr'), resized)

if __name__ == '__main__':
    imageio.plugins.freeimage.download() # Download needed plugins

    args = parse_args()
    folders = sorted(glob.glob(os.path.join(args.root_dir, '*', '')))
    not_working_folders = []
    for folder in folders:
        if os.path.exists(os.path.join(folder, 'env_maps.npy')):
            proc_folder(args, folder)
        else:
            print('Not working!')
            not_working_folders.append(folder)
    print(not_working_folders)