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
    parser.add_argument('--num_images', type=int,
                        default=20,
                        help='number of source images to render (x3 for total images)')
    
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


def proc_src(input_folder, output_folder, out_rot_folder, num_images):
    '''Collect source views
    '''
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(out_rot_folder, exist_ok=True)

    index = 0

    shapes = []
    self_rot_shapes = []

    for i in range(num_images):
        ############# Process Source Images #############
        filename = f'{index:03d}'
        # Read RGB
        rgb_path = os.path.join(input_folder, 'rgb', f'{filename}.exr')
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)          
        rgb_max = np.random.uniform(0.9, 1.1) * np.max(rgb)
        rgb /= rgb_max
        srgb = np.clip(linear2srgb(rgb), 0.0, 1.0)
        srgb = (srgb * 255.).astype(np.uint8)

        # Process RGB
        out_rgb = os.path.join(output_folder, f'{i:03d}_image.jpg')
        cv2.imwrite(out_rgb, srgb)
        shape = srgb.shape[:2]
        shapes.append(shape)

        # Process env map
        env_path = os.path.join(input_folder, 'envmap', f'{filename}_light.hdr')
        env_map = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
        env_map /= rgb_max
        out_path = os.path.join(output_folder, f'{i:03d}_light.hdr')
        cv2.imwrite(out_path, env_map)

        env_path = os.path.join(input_folder, 'envmap', f'{filename}_lightcam.hdr')
        env_map = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
        env_map /= rgb_max
        out_path = os.path.join(output_folder, f'{i:03d}_lightcam.hdr')
        cv2.imwrite(out_path, env_map)

        # Process mask
        mask_path = os.path.join(input_folder, 'mask', f'{filename}_0001.exr')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = (mask * 255.).astype(np.uint8)
        out_mask = os.path.join(output_folder, f'{i:03d}_mask.jpg')
        cv2.imwrite(out_mask, mask)

        # Process depth
        depth_path = os.path.join(input_folder, 'depth', f'{filename}_0001.exr')
        dst = os.path.join(output_folder, f'{i:03d}_depth.exr')
        shutil.copy2(depth_path, dst)

        ############# Process Target Self Rotation Images #############
        filename = f'{index + num_images:03d}'
        # Process RGB
        rgb_path = os.path.join(input_folder, 'rgb', f'{filename}.exr')
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        rgb /= rgb_max
        srgb = np.clip(linear2srgb(rgb), 0.0, 1.0)
        srgb = (srgb * 255.).astype(np.uint8)
        out_rgb = os.path.join(out_rot_folder, f'{i:03d}_image.jpg')
        cv2.imwrite(out_rgb, srgb)
        shape = srgb.shape[:2]
        self_rot_shapes.append(shape)

        # Process env map
        env_path = os.path.join(input_folder, 'envmap', f'{filename}_light.hdr')
        env_map = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
        env_map /= rgb_max
        out_path = os.path.join(out_rot_folder, f'{i:03d}_light.hdr')
        cv2.imwrite(out_path, env_map)

        env_path = os.path.join(input_folder, 'envmap', f'{filename}_lightcam.hdr')
        env_map = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
        env_map /= rgb_max
        out_path = os.path.join(out_rot_folder, f'{i:03d}_lightcam.hdr')
        cv2.imwrite(out_path, env_map)

        # Process mask
        mask_path = os.path.join(input_folder, 'mask', f'{filename}_0001.exr')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = (mask * 255.).astype(np.uint8)
        out_mask = os.path.join(out_rot_folder, f'{i:03d}_mask.jpg')
        cv2.imwrite(out_mask, mask)

        # Process depth
        depth_path = os.path.join(input_folder, 'depth', f'{filename}_0001.exr')
        dst = os.path.join(out_rot_folder, f'{i:03d}_depth.exr')
        shutil.copy2(depth_path, dst)

        # Process txt
        txt_path = os.path.join(input_folder, 'txt', f'{i:03d}.txt')
        dst = os.path.join(out_rot_folder, f'{i:03d}_light.txt')
        with open(txt_path, 'r') as f:
            deg = float(f.read())
        with open(dst, 'w') as f:
            px = deg / 360 * 16
            f.write(f"{px}")


        index += 1

    ################### Source Views ###################
    # Process camera params
    intrinsics = np.load(os.path.join(input_folder, 'cam_ints.npy'))
    extrinsics = np.load(os.path.join(input_folder, 'cam_exts.npy'))
    extrinsics = [get_3x4_RT_matrix_from_blender(x) for x in extrinsics]
    extrinsics = np.array(extrinsics)

    # Get set params
    intrinsics = np.array(intrinsics[:index])
    extrinsics = np.array(extrinsics[:index])

    shapes = np.array(shapes).reshape([args.num_images, 2])

    np.savez(
        os.path.join(output_folder, 'cameras.npz'),
        shapes=shapes,
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )

    ################### Target Self Rotation ###################
    # Process camera params
    intrinsics = np.load(os.path.join(input_folder, 'cam_ints.npy'))
    extrinsics = np.load(os.path.join(input_folder, 'cam_exts.npy'))
    extrinsics = [get_3x4_RT_matrix_from_blender(x) for x in extrinsics]
    extrinsics = np.array(extrinsics)

    # Get set params
    intrinsics = np.array(intrinsics[index:index+num_images])
    extrinsics = np.array(extrinsics[index:index+num_images])

    self_rot_shapes = np.array(self_rot_shapes).reshape([num_images, 2])

    np.savez(
        os.path.join(out_rot_folder, 'cameras.npz'),
        shapes=self_rot_shapes,
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )


def proc_tgt(input_folder, output_folder, indices):
    os.makedirs(output_folder, exist_ok=True)
 
    shapes = []
    for i, index in enumerate(indices):
        filename = f'{index:03d}'
        
        # Process RGB
        rgb_path = os.path.join(input_folder, 'rgb', f'{filename}.exr')
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        rgb_max = np.random.uniform(0.9, 1.1) * np.max(rgb)
        rgb /= rgb_max
        srgb = np.clip(linear2srgb(rgb), 0.0, 1.0)
        srgb = (srgb * 255.).astype(np.uint8)
        out_rgb = os.path.join(output_folder, f'{i:03d}_image.jpg')
        cv2.imwrite(out_rgb, srgb)
        shape = srgb.shape[:2]
        shapes.append(shape)

        # Process env map
        env_path = os.path.join(input_folder, 'envmap', f'{filename}_light.hdr')
        env_map = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
        env_map /= rgb_max
        out_path = os.path.join(output_folder, f'{i:03d}_light.hdr')
        cv2.imwrite(out_path, env_map)

        env_path = os.path.join(input_folder, 'envmap', f'{filename}_lightcam.hdr')
        env_map = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
        env_map /= rgb_max
        out_path = os.path.join(output_folder, f'{i:03d}_lightcam.hdr')
        cv2.imwrite(out_path, env_map)

        # Process mask
        mask_path = os.path.join(input_folder, 'mask', f'{filename}_0001.exr')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = (mask * 255.).astype(np.uint8)
        out_mask = os.path.join(output_folder, f'{i:03d}_mask.jpg')
        cv2.imwrite(out_mask, mask)

        # Process depth
        depth_path = os.path.join(input_folder, 'depth', f'{filename}_0001.exr')
        dst = os.path.join(output_folder, f'{i:03d}_depth.exr')
        shutil.copy2(depth_path, dst)

    # Process camera params
    intrinsics = np.load(os.path.join(input_folder, 'cam_ints.npy'))[indices]
    extrinsics = np.load(os.path.join(input_folder, 'cam_exts.npy'))[indices]
    extrinsics = [get_3x4_RT_matrix_from_blender(x) for x in extrinsics]
    extrinsics = np.array(extrinsics)
    shapes = np.array(shapes)

    np.savez(
        os.path.join(output_folder, 'cameras.npz'),
        shapes=shapes,
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )

if __name__ == '__main__':
    args = parse_args()
    folders = sorted(glob.glob(os.path.join(args.root_dir, '*', '')))
    
    for folder in folders:
        out_folder = os.path.join(args.out_dir, os.path.split(folder[:-1])[-1], 'source_image')
        out_rot_folder = os.path.join(args.out_dir, os.path.split(folder[:-1])[-1], 'target_image_rotation')
        proc_src(folder, out_folder, out_rot_folder, args.num_images)

        out_folder = os.path.join(args.out_dir, os.path.split(folder[:-1])[-1], 'target_image')
        s = args.num_images * 2
        e = args.num_images * 3
        indices = [x for x in range(s, e)]
        proc_tgt(folder, out_folder, indices)