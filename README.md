# FaceScape Rendering Script
Blender rendering script for FaceScape dataset accompanying paper ["NeLF: Neural Light-transport Field for Portrait View Synthesis and Relighting"](https://github.com/ken2576/nelf)


### Requirements

* Blender

* PyTorch & Kornia - for downsampling

* numpy

* OpenCV-Python

* matplotlib

### Usage

The following steps denote how to generate training data for NeLF.

0. Prepare dataset. Download TU-Model from [FaceScape dataset](https://facescape.nju.edu.cn/). Extract them to a desired folder.
    Example data structure:
    ```
    [path_to_model_folder] --- 1 --- dpmap
                            |     |- models_reg
                            |- 2
                            |- 3
                            ...
    ```

1. Remove the red cap texture in FaceScape dataset by running:
    ```
    python remove_hat.py --model_folder [path_to_model_folder] --out_folder [path_to_output_model_folder]
    ```

2. Prepare environment map. There are some free environment maps available online. For example, the [Laval Indoor HDR Dataset](http://indoor.hdrdb.com/). Download and extract them to a desired location.

    Example data structure:
    ```
    [path_to_envmap_folder] --- 0.hdr
                             |- 1.hdr
                             ...
    ```

3. (Optional) Sometimes the environment maps are too dimmed. They can be adjusted with a normalization script:

    ```
    python normalize_env.py --folder [path_to_envmap_folder] --out_dir [path_to_output_envmap_folder]
    ```

4. Randomly rotate the environment map for self rotation training
    ```
    python random_rotate.py --root_dir [path_to_envmap_folder] --out_dir [path_to_rotated_envmap_folder]
    ```

5. Set up config files. Change the paths in ```configs/default.txt``` to your corresponding folders.

6. Render models with Blender

    ```
    python batch_render_default.py --config configs/default.txt
    ```

7. Prepare downscaled environment map for the outputs

    ```
    python prepare_env_map.py --root_dir [root_folder_of_output]
    ```

8. Collect dataset

    ```
    python collect_data_default.py --root_dir [root_folder_of_output] \
    --out_dir [output_folder]
    ```


(Optional) Generate additional data for IBRNet and SIPR

1. Set up config files. Change the paths in ```configs/relight.txt``` to your corresponding folders for SIPR relighting training data and in ```configs/uniform.txt``` for IBRNet view synthesis training data.

2. Render models with Blender

    For relighting:
    ```
    python batch_render_relight.py --config configs/relight.txt
    ```
    For view synthesis:
    ```
    python batch_render_uniform.py --config configs/uniform.txt
    ```

3. Prepare downscaled environment map for the outputs

    ```
    python prepare_env_map.py --root_dir [root_folder_of_output]
    ```

4. Collect dataset

    For relighting:
    ```
    python collect_data_relight.py --root_dir [root_folder_of_output] \
    --out_dir [output_folder]
    ```

    For view synthesis:
    ```
    python collect_data_uniform.py --root_dir [root_folder_of_output] \
    --out_dir [output_folder]
    ```