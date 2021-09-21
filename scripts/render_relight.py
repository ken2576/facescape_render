import os
import glob
import argparse
import json
import sys
import shutil

from pose_utils import gen_random_poses, gen_random_frontal_pose, gen_multiview_poses
import numpy as np
import bpy
import utils

from mathutils import Matrix, Vector

def get_fov_from_pose(pose, radius, obj_dim):
    rad = np.arctan2(obj_dim[1]/2., radius) * 2
    deg = np.rad2deg(rad)
    return deg

def get_intrinsics(args, fov):
    angle = np.deg2rad(fov)
    aspect_ratio = args.img_wh[0] / args.img_wh[1]
    K = np.zeros((3,3), dtype=np.float32)
    K[0][0] = args.img_wh[0] / 2 / np.tan(angle / 2)
    K[1][1] = args.img_wh[1] / 2. / np.tan(angle / 2) * aspect_ratio
    K[0][2] = args.img_wh[0] / 2.
    K[1][2] = args.img_wh[1] / 2.
    K[2][2] = 1.
    return K

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT

def read_hdr(folder):
    hdr_list = sorted(glob.glob(os.path.join(folder, '*.hdr')))
    return hdr_list

def load_envmap(path):
    # Load envmap
    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = bpy.context.scene.world.node_tree
    enode = node_tree.nodes.new("ShaderNodeTexEnvironment")
    enode.image = bpy.data.images.load(path)
    node_tree.links.new(enode.outputs['Color'],
                node_tree.nodes['Background'].inputs['Color'])
    return enode

def load_face(folder, expression, scale):
    obj_path = os.path.join(folder, 'models_reg', expression + '.obj')
    displacement_path = os.path.join(folder, 'dpmap', expression + '.png')

    # Create object
    bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='-Z', axis_up='Y')
    mesh_object = bpy.context.selected_objects[0]
    utils.setSmooth(mesh_object, 2)
    bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center='BOUNDS')
    bpy.ops.transform.resize(value=(scale, scale, scale))

    if len(glob.glob(displacement_path)) != 0:
        heightTex = bpy.data.textures.new('Displacement', type = 'IMAGE')
        heightTex.image = bpy.data.images.load(displacement_path)
        dispMod = mesh_object.modifiers.new("Displace", type='DISPLACE')
        dispMod.texture = heightTex
        dispMod.texture_coords = 'UV'

    mesh_object.material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs['Specular'].default_value=0.2
    mesh_object.material_slots[0].material.node_tree.nodes["Principled BSDF"].inputs['Roughness'].default_value=0.5

    return mesh_object

def get_rotated_env_path(ori_env_path, rotated_folder):
    env_id = os.path.split(ori_env_path)[-1][:-4]
    rot_paths = glob.glob(os.path.join(rotated_folder, f'{env_id}_*.exr'))
    new_env_path = np.random.choice(rot_paths, 1)[0]
    new_env_txt = new_env_path[:-4] + '.txt'
    return new_env_path, new_env_txt

def parse_args(argv):
    # Parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--obj_path', type=str, metavar='PATH',
                        help='path to the object file')
    parser.add_argument('--env_folder', type=str, metavar='PATH',
                        default=None,
                        help='path to the environment map folder')
    parser.add_argument('--rotated_folder', type=str, metavar='PATH',
                        default=None,
                        help='path to the rotated environment map folder')
    parser.add_argument('--out_dir', type=str, metavar='PATH',
                        help='path to the output directory')
    parser.add_argument('--num_images', type=int,
                        default=20,
                        help='number of source images to render (x3 for total images)')
    parser.add_argument('--theta_max', type=float, default=30.0,
                        help='maximum azimuthal angle change')
    parser.add_argument('--phi_max', type=float, default=30.0,
                        help='maximum elevational angle change')
    parser.add_argument('--rmin', type=float, default=1000.0,
                        help='minimum radius of the camera position')
    parser.add_argument('--rmax', type=float, default=2000.0,
                        help='maximum radius of the camera position')
    parser.add_argument('--img_wh', type=int, nargs='+', default=[512, 512],
                        help='image resolution (width, height)')
    parser.add_argument('--num_samples', type=int, default=128,
                        help='number of samples for ray tracing')
    parser.add_argument('--cpu', action='store_true',
                        help='use CPU to render')
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                        help="GPU id for render")
    return parser.parse_args(argv)

if __name__ == '__main__':
    argv = sys.argv
    try:
        index = argv.index("--") + 1
    except ValueError:
        index = len(argv)

    argv = argv[index:]
    print(argv)

    args = parse_args(argv)

    obj_path = args.obj_path
    out_dir = args.out_dir

    poses = []
    radii = []

    ############## Get Poses ##############
    poses, radii = gen_random_poses(args.rmin,
                                    args.rmax,
                                    args.theta_max,
                                    args.phi_max,
                                    1)

    # Remove all elements
    utils.removeAll()

    # Create camera
    target = utils.target()

    # Set cursor to (0, 0, 0)
    bpy.context.scene.cursor.location = (0, 0, 0)

    # Background
    bpy.context.scene.render.dither_intensity = 0.0
    bpy.context.scene.render.film_transparent = True
    
    # Set image format
    bpy.context.scene.render.image_settings.color_depth = str(16)
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'  

    # Set pass
    bpy.context.scene.view_layers["View Layer"].use_pass_normal = True

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)

    render_layers = tree.nodes.new('CompositorNodeRLayers')

    # Create output folder
    rgb_dir = os.path.join(out_dir, 'rgb')
    os.makedirs(rgb_dir, exist_ok=True)
    depth_dir = os.path.join(out_dir, 'depth')
    os.makedirs(depth_dir, exist_ok=True)
    normal_dir = os.path.join(out_dir, 'normal')
    os.makedirs(normal_dir, exist_ok=True)
    mask_dir = os.path.join(out_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    txt_dir = os.path.join(out_dir, 'txt')
    os.makedirs(txt_dir, exist_ok=True)

    # Set output path
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    depth_file_output.base_path = depth_dir

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
    normal_file_output.base_path = normal_dir

    mask_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    mask_file_output.label = 'Mask Output'
    links.new(render_layers.outputs['Alpha'], mask_file_output.inputs[0])
    mask_file_output.base_path = mask_dir

    # Set up renderer params
    scene = bpy.data.scenes["Scene"]
    scene.render.engine = 'CYCLES'
    scene.render.use_motion_blur = False
    if args.cpu:
        scene.cycles.device = 'CPU'
    else:
        scene.cycles.device = 'GPU'
    scene.render.tile_x = 256
    scene.render.tile_y = 256

    scene.render.film_transparent = True
    scene.view_layers[0].cycles.use_denoising = True

    scene.cycles.samples = args.num_samples

    # GPU config
    if not args.cpu:
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'
        cycles_prefs.get_devices()
        for di, device in enumerate(cycles_prefs.devices):
            device.use = (di == args.gpu_id)

    # Load face model
    scale = np.random.uniform(0.9, 1.1)
    mesh_object = load_face(obj_path, '18_eye_closed', scale)

    # Save object dimension data
    obj_dim = mesh_object.dimensions
    bbox_corners = [mesh_object.matrix_world @ Vector(corner) for corner in mesh_object.bound_box]

    obj_dict = dict(obj_dim=np.array(obj_dim).tolist(),
                    bbox=np.array(bbox_corners).tolist())

    with open(os.path.join(out_dir, 'commandline_args.txt'), 'w') as f:
        json.dump([args.__dict__, obj_dict], f, indent=2)

    # Load environment map
    env_paths = glob.glob(os.path.join(args.env_folder, '*.hdr'))
    env_paths = np.array(env_paths)

    cam_extrinsics = np.zeros((args.num_images*3, 4, 4))
    cam_intrinsics = np.zeros((args.num_images*3, 3, 3))
    rendered_env = []

    img_idx = 0

    pose = poses[0]
    radius = radii[0]
   
    #################### Render source views ####################
    for i in range(args.num_images):
        # Load env map
        env_path = np.random.choice(env_paths)
        load_envmap(env_path)

        # Set up camera
        fov = get_fov_from_pose(pose, radius, obj_dim) * 1.1 # slight margin to prevent out of FOV
        K = get_intrinsics(args, fov)
        camera = utils.camera(pose, target, fov)
        
        bpy.context.view_layer.update()
        
        RT = get_3x4_RT_matrix_from_blender(camera)
        
        temp = camera.matrix_world
        cam_extrinsics[img_idx] = np.asarray(utils.listify_matrix(temp))
        cam_intrinsics[img_idx] = K

        # Render scene
        utils.render(out_dir, f'{img_idx:03d}',
                    depth_file_output,
                    normal_file_output,
                    mask_file_output,
                    args.img_wh[0], args.img_wh[1])

        rendered_env.append(env_path)
        img_idx += 1
        

    #################### Render target self rotation ####################
    for i in range(args.num_images):
        ori_env_path = rendered_env[i]
        env_path, env_txt = get_rotated_env_path(ori_env_path, args.rotated_folder)
        shutil.copy2(env_txt, os.path.join(txt_dir, f'{i:03d}.txt'))
        load_envmap(env_path)

        # Set up camera
        fov = get_fov_from_pose(pose, radius, obj_dim) * 1.1 # slight margin to prevent out of FOV
        K = get_intrinsics(args, fov)
        camera = utils.camera(pose, target, fov)
        
        bpy.context.view_layer.update()
        
        RT = get_3x4_RT_matrix_from_blender(camera)
        
        temp = camera.matrix_world
        cam_extrinsics[img_idx] = np.asarray(utils.listify_matrix(temp))
        cam_intrinsics[img_idx] = K

        # Render scene
        utils.render(out_dir, f'{img_idx:03d}',
                    depth_file_output,
                    normal_file_output,
                    mask_file_output,
                    args.img_wh[0], args.img_wh[1])

        rendered_env.append(env_path)
        img_idx += 1

    #################### Render target views ####################
    for i in range(args.num_images):
        # Load env map
        env_path = np.random.choice(env_paths)
        load_envmap(env_path)

        # Set up camera
        fov = get_fov_from_pose(pose, radius, obj_dim) * 1.1 # slight margin to prevent out of FOV
        K = get_intrinsics(args, fov)
        camera = utils.camera(pose, target, fov)
        
        bpy.context.view_layer.update()
        
        RT = get_3x4_RT_matrix_from_blender(camera)
        
        temp = camera.matrix_world
        cam_extrinsics[img_idx] = np.asarray(utils.listify_matrix(temp))
        cam_intrinsics[img_idx] = K

        # Render scene
        utils.render(out_dir, f'{img_idx:03d}',
                    depth_file_output,
                    normal_file_output,
                    mask_file_output,
                    args.img_wh[0], args.img_wh[1])

        rendered_env.append(env_path)
        img_idx += 1

    rendered_env = np.array(rendered_env)

    np.save(os.path.join(out_dir, 'cam_ints.npy'), cam_intrinsics)
    np.save(os.path.join(out_dir, 'cam_exts.npy'), cam_extrinsics)
    np.save(os.path.join(out_dir, 'env_maps.npy'), rendered_env)