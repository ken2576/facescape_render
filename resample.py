'''Resample environment map
'''

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import kornia

def downsample(img, new_hw=(-1, -1)):
    '''Downsample images with Gaussian pyramid, each is 2x downsample

    Args:
        img: RGB image [batch, #channels, height, width]
        new_hw: desired new dimensions (need to be the same aspect ratio)
    Returns:
        Downsampled image
    '''
    src_h, src_w = img.shape[-2:]
    num_levels = int(np.log2(src_h / new_hw[0]))

    downscaled = img
    for x in range(num_levels-1):
        ks = 2 ** (num_levels + 2 - x) - 1
        downscaled = kornia.filters.gaussian_blur2d(downscaled, (ks, ks), (3, 3))
        downscaled = F.interpolate(downscaled, scale_factor=0.5, recompute_scale_factor=False)

    downscaled = kornia.filters.gaussian_blur2d(downscaled, (3, 3), (1.5, 1.5))
    downscaled = F.interpolate(downscaled, new_hw)
    return downscaled


def sphere_coords(h, w, device='cpu', is_homogenous=True):
    '''Create camera coordinates of a panorama
    
    Args:
        h: image height
        w: image width
        device: which device to use 'cpu' or 'cuda'
        is_homogenous: return in homogenous coordinates
    Returns:
        Camera coordinates of the sphere with radius 1
        [height, width, 3 (4 if homogenous)]
    '''
    # Creating coordinates phi, theta corresponding to y, x
    xs = torch.linspace(0, w-1, steps=w, device=device) + 0.5
    ys = torch.linspace(0, h-1, steps=h, device=device) + 0.5
    new_y, new_x = torch.meshgrid(ys/h, xs/w)

    # horizontal/azimuthal
    theta = 2 * np.pi * (new_x - 0.5)
    # elevation
    phi = np.pi * (new_y-0.5)
    
    cam_coords = torch.stack([
        torch.cos(phi) * torch.sin(theta),
        torch.sin(phi),
        torch.cos(phi) * torch.cos(theta)
    ], -1)


    if is_homogenous:
        cam_coords = torch.cat([cam_coords, torch.ones_like(cam_coords[..., :1])], 2)

    return cam_coords

def cartesian_to_polar(coord):
    '''Convert world coordinates to equirectangular coordinates
    Args:
        coord: cartesian camera coordiantes [..., 4]
    Returns:
        Polar coordinates [..., 2]
    '''
    theta = torch.atan2(coord[..., 0], coord[..., 2]) / np.pi
    norm = (coord[..., 0]**2 + coord[..., 2]**2)**0.5
    phi = torch.atan2(coord[..., 1], norm) / np.pi * 2
    return torch.stack([theta, phi], -1)

def pano_inverse_warp(img, rotation, h=-1, w=-1):
    '''Inverse warp a source environment map to the target pose

    Args:
        img: source image [batch, #channels, height, width]
        rotation: rotation matrix from target to source [batch, 3, 3]
    Returns:
        Warped image [batch, #channels, height, width]
    '''
    if h == -1 or w == -1:
        h, w = img.shape[-2:]
    b = img.shape[0]

    # Generate image coordinates for target camera
    cam_coord = sphere_coords(h, w, device=img.device).view([-1, 4, 1])
    cam_coord = cam_coord.unsqueeze(0).repeat([b, 1, 1, 1])

    # Create transformation matrix
    trnsf = torch.eye(4, device=img.device)[None, ...].repeat([b, 1, 1])
    trnsf[..., :3, :3] = rotation

    # Convert to another camera's coordinates
    new_cam_coord = torch.matmul(trnsf.unsqueeze(1), cam_coord)
    pix_coord = cartesian_to_polar(new_cam_coord.squeeze(-1))
    pix_coord = pix_coord.reshape([b, h, w, 2])
 
    # Sample from the source image
    warped = F.grid_sample(img, pix_coord, padding_mode='reflection', align_corners=True)
    return warped

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(matrix_world):
    def decompose(mat):
        loc = mat[:3, -1]
        rot = mat[:3, :3]
        return loc, rot
    # bcam stands for blender camera
    R_bcam2cv = np.array(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = decompose(matrix_world)
    # print(matrix_world)
    R_world2bcam = rotation.T

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # print(R_world2bcam)
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = np.concatenate([R_world2cv, T_world2cv[:, None]], 1)
    world_fix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    return RT# @ world_fix