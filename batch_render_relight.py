import os
import glob
import re
import configargparse
import subprocess
import datetime

def parse_args():
    # Parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument('--model_folder', type=str, metavar='PATH',
                        required=True,
                        help='path to the folder containing models')
    parser.add_argument('--env_folder', type=str, metavar='PATH',
                        required=True,
                        help='path to the folder containing env maps')
    parser.add_argument('--rotated_folder', type=str, metavar='PATH',
                        default=None,
                        help='path to the rotated environment map folder')
    parser.add_argument('--out_folder', type=str, metavar='PATH',
                        required=True,
                        help='path to the rendering output folder')
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
    parser.add_argument('-s', '--start', type=int, default=None,
                        help="starting point")
    parser.add_argument('-e', '--end', type=int, default=None,
                        help="ending point")
    return parser.parse_args()

args = parse_args()

models = sorted(glob.glob(os.path.join(args.model_folder, '*')),
    key=lambda f: int(re.sub('\D', '', f)))
models_to_render = models[args.start:args.end]
print(f'Models to render: {len(models_to_render)}')
print(f"Current range: [{args.start}, {args.end}]")

for idx, model in enumerate(models_to_render):
    print(f"Current path: {model}")
    print(f"Timestamp: {datetime.datetime.now()}")

    start = args.start if args.start else 0
    out_dir = os.path.join(args.out_folder, f'sub{start+idx+1:03d}')

    blender_args = [
        'blender', '-b', '-P',
        'run_script_relight.py',
        '--', '--obj_path', model,
        '--env_folder', args.env_folder,
        '--rotated_folder', args.rotated_folder,
        '--out_dir', out_dir,
        '--num_images', f'{args.num_images}',
        '--theta_max', f'{args.theta_max:.3f}',
        '--phi_max', f'{args.phi_max:.3f}',
        '--rmin', f'{args.rmin:.3f}',
        '--rmax', f'{args.rmax:.3f}',
        '--img_wh', f'{args.img_wh[0]}', f'{args.img_wh[1]}',
        '--num_samples', f'{args.num_samples}',
    ]
    if args.cpu:
        blender_args += ['--cpu']
    subprocess.check_output(blender_args, universal_newlines=True)
