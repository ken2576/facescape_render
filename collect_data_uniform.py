import os
import glob
import shutil
import argparse
import numpy as np
import cv2



def parse_args():
    # Parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', type=str, metavar='PATH',
                        help='path to data root directory')
    parser.add_argument('--out_dir', type=str, metavar='PATH',
                        help='path to output directory')
    parser.add_argument('--num_sets', type=int,
                        default=10,
                        help='number of sets')
    parser.add_argument('--num_views', type=int,
                        default=5,
                        help='number of source views')
    parser.add_argument('--num_tgts', type=int,
                        default=30,
                        help='number of target views')
    return parser.parse_args()

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
    return RT

def linear2srgb(linear):
    srgb = np.float32(linear)
    return np.where(
        srgb <= 0.0031308,
        srgb * 12.92,
        1.055 * np.power(srgb, 1.0 / 2.4) - 0.055
    )

def proc_images(input_folder, output_folder, num_sets, num_views, num_tgts):
    '''Collect source views
    '''
    src_folder = os.path.join(output_folder, 'source_image')
    tgt_folder = os.path.join(output_folder, 'target_image')
    os.makedirs(src_folder, exist_ok=True)
    os.makedirs(tgt_folder, exist_ok=True)
 
    index = 0
    shapes = []
    for i in range(num_sets):
        # Read RGB
        rgb_paths = [os.path.join(input_folder, 'rgb', f'{x:03d}.exr') for x in range(i*num_views, (i+1)*num_views)]
        rgbs = [cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED) for rgb_path in rgb_paths]   
        rgbs = np.stack(rgbs, 0)
        if i == 0:
            rgb_mean = np.mean(rgbs) / 0.1
        rgbs /= rgb_mean
        srgb = np.clip(linear2srgb(rgbs), 0.0, 1.0)
        srgb = (srgb * 255.).astype(np.uint8)

        for j in range(num_views):
            filename = f'{index:03d}'

            # Process RGB
            out_rgb = os.path.join(src_folder, f'{i:03d}_{j:03d}_image.jpg')
            cv2.imwrite(out_rgb, srgb[j])
            shape = srgb[j].shape[:2]
            shapes.append(shape)

            # Process mask
            mask_path = os.path.join(input_folder, 'mask', f'{filename}_0001.exr')
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = (mask * 255.).astype(np.uint8)
            out_mask = os.path.join(src_folder, f'{i:03d}_{j:03d}_mask.jpg')
            cv2.imwrite(out_mask, mask)

            # Process depth
            depth_path = os.path.join(input_folder, 'depth', f'{filename}_0001.exr')
            dst = os.path.join(src_folder, f'{i:03d}_{j:03d}_depth.exr')
            shutil.copy2(depth_path, dst)

            index += 1

    # Process camera params
    intrinsics = np.load(os.path.join(input_folder, 'cam_ints.npy'))
    extrinsics = np.load(os.path.join(input_folder, 'cam_exts.npy'))
    extrinsics = [get_3x4_RT_matrix_from_blender(x) for x in extrinsics]
    extrinsics = np.array(extrinsics)

    # Get set params
    intrinsics = np.array(intrinsics[:index]).reshape([args.num_sets, args.num_views, 3, 3])
    extrinsics = np.array(extrinsics[:index]).reshape([args.num_sets, args.num_views, 3, 4])

    shapes = np.array(shapes).reshape([args.num_sets, args.num_views, 2])

    np.savez(
        os.path.join(src_folder, 'cameras.npz'),
        shapes=shapes,
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )


    shapes = []
    for i in range(num_tgts):
        filename = f'{index:03d}'
        
        # Process RGB
        rgb_path = os.path.join(input_folder, 'rgb', f'{filename}.exr')
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        rgb /= rgb_mean
        srgb = np.clip(linear2srgb(rgb), 0.0, 1.0)
        srgb = (srgb * 255.).astype(np.uint8)
        out_rgb = os.path.join(tgt_folder, f'{i:03d}_image.jpg')
        cv2.imwrite(out_rgb, srgb)
        shape = srgb.shape[:2]
        shapes.append(shape)

        # Process mask
        mask_path = os.path.join(input_folder, 'mask', f'{filename}_0001.exr')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = (mask * 255.).astype(np.uint8)
        out_mask = os.path.join(tgt_folder, f'{i:03d}_mask.jpg')
        cv2.imwrite(out_mask, mask)

        # Process depth
        depth_path = os.path.join(input_folder, 'depth', f'{filename}_0001.exr')
        dst = os.path.join(tgt_folder, f'{i:03d}_depth.exr')
        shutil.copy2(depth_path, dst)

        index += 1

    # Process camera params
    intrinsics = np.load(os.path.join(input_folder, 'cam_ints.npy'))[num_sets*num_views:]
    extrinsics = np.load(os.path.join(input_folder, 'cam_exts.npy'))[num_sets*num_views:]
    extrinsics = [get_3x4_RT_matrix_from_blender(x) for x in extrinsics]
    extrinsics = np.array(extrinsics)
    shapes = np.array(shapes)

    np.savez(
        os.path.join(tgt_folder, 'cameras.npz'),
        shapes=shapes,
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )


if __name__ == '__main__':
    args = parse_args()
    folders = sorted(glob.glob(os.path.join(args.root_dir, '*', '')))
    
    for folder in folders:
        out_folder = os.path.join(args.out_dir, os.path.split(folder[:-1])[-1])
        proc_images(folder, out_folder, args.num_sets, args.num_views, args.num_tgts)