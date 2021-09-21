import os
import glob
import shutil
import argparse
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, metavar='PATH',
                        required=True,
                        help='path to the folder containing models')
    parser.add_argument('--out_folder', type=str, metavar='PATH',
                        required=True,
                        help='path to the rendering output folder')
    return parser.parse_args()


def remove_hat(image):
    """Input OpenCV BGR image and remove red hat
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_channels = cv2.split(hsv)
    mask = (hsv_channels[0] > 150) | (hsv_channels[1] > 170)
    h, w = mask.shape
    mask[h//4:, w*3//8:w*5//8] = 0
    mask[h//2:] = 0

    specular = (1-mask) * 0.2 * 255
    specular = specular.astype(np.uint8)

    hair_color = np.ones([h, w, 3], dtype=np.uint8) * np.array([10, 90, 15], dtype=np.uint8)[None, None, :]
    new_hsv = np.where(mask[:, :, None], hair_color, hsv)
    # new_hsv = np.stack(hsv_channels, -1)
    new_rgb = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
    return new_rgb, specular

def proc_model(model_folder, out_folder):
    image_path = os.path.join(model_folder, 'models_reg', '18_eye_closed.jpg')
    if not os.path.exists(image_path):
        print(f'{model_folder}: no texture found!')
        return
    img = cv2.imread(image_path)
    new_img, specular = remove_hat(img)

    src_dpmap = os.path.join(model_folder, 'dpmap', '18_eye_closed.png')
    if os.path.exists(src_dpmap):
        new_dp = os.path.join(out_folder, 'dpmap')
        os.makedirs(new_dp, exist_ok=True)
        dst_dpmap = os.path.join(new_dp, '18_eye_closed.png')
        shutil.copy2(src_dpmap, dst_dpmap)

    new_model = os.path.join(out_folder, 'models_reg')
    os.makedirs(new_model, exist_ok=True)
    src_model = os.path.join(model_folder, 'models_reg', '18_eye_closed.obj')
    dst_model = os.path.join(new_model, '18_eye_closed.obj')
    shutil.copy2(src_model, dst_model)

    src_model = os.path.join(model_folder, 'models_reg', '18_eye_closed.obj.mtl')
    dst_model = os.path.join(new_model, '18_eye_closed.obj.mtl')
    shutil.copy2(src_model, dst_model)

    cv2.imwrite(os.path.join(new_model, '18_eye_closed.jpg'), new_img)
    cv2.imwrite(os.path.join(new_model, '18_eye_closed_specular.png'), specular)


if __name__ == '__main__':

    args = parse_args()
    models = sorted(glob.glob(os.path.join(args.model_folder, '*')),
        key=lambda f: int(re.sub('\D', '', f)))

    for model in models:
        print(model)

        model_name = os.path.split(model)[-1]
        out_folder = os.path.join(args.out_folder, model_name)
        proc_model(model, out_folder)