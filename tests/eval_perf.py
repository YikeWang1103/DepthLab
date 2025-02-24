import sys
import os
import argparse
import json
import cv2
import copy

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from types import SimpleNamespace

# 获取当前工作目录
current_dir = os.getcwd()
print(f"current_dir: {current_dir}")

# 将当前工作目录添加到 sys.path
if current_dir not in sys.path:
    sys.path.append(current_dir)

from tests.depthlab_wrapper import *
from tests.tester import *

def arg_parse():
    parser = argparse.ArgumentParser(
        description="Performance evaluation of DepthLab based on Fusionride datasets!"
    )

    parser.add_argument(
        '--dumped_path', type=str, required=True
    )

    parser.add_argument(
        '--sensor_config', type=str, required=True
    )

    parser.add_argument(
        '--model_hyper_param_config', type=str, required=True
    )

    parser.add_argument(
        '--refine',
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        '--output_path', type=str, required=True
    )

    args = parser.parse_args()
    return args

def get_hyper_param(hyper_param_path):
    with open(hyper_param_path, 'r') as f:
        hyper_param_js = json.load(f)

    hyper_param = SimpleNamespace(
        denoise_steps = int(hyper_param_js["denoise_steps"]),
        processing_res = int(hyper_param_js["processing_res"]),
        normalize_scale = hyper_param_js["normalize_scale"],
        strength = hyper_param_js["strength"],
        seed = int(hyper_param_js["seed"]),
        pretrained_model_name_or_path = hyper_param_js["pretrained_model_name_or_path"],
        image_encoder_path = hyper_param_js["image_encoder_path"],
        denoising_unet_path = hyper_param_js["denoising_unet_path"],
        reference_unet_path = hyper_param_js["reference_unet_path"],
        mapping_path = hyper_param_js["mapping_path"],
        blend = hyper_param_js["blend"]
    )

    return hyper_param

def get_sensor_calib(sensor_config):
    with open(sensor_config, 'r') as f:
        sensor_config_js = json.load(f)

    sensor_calib = SimpleNamespace(
        RADAR_FRONT_2_EGO = np.array(sensor_config_js["RADAR_FRONT_2_EGO"], np.float32),
        LIDAR_TOP_2_EGO = np.array(sensor_config_js["LIDAR_TOP_2_EGO"], np.float32),
        CAM_FRONT_2_EGO = np.array(sensor_config_js["CAM_FRONT_2_EGO"], np.float32),
        CAM_FRONT_INTRINSICS = np.array(sensor_config_js["CAM_FRONT_INTRINSICS"], np.float32),
        RESOLUTION = sensor_config_js["RESOLUTION"]
    )

    return sensor_calib

if __name__ == '__main__':
    args = arg_parse()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    hyper_param = get_hyper_param(args.model_hyper_param_config)

    sensor_calib = get_sensor_calib(args.sensor_config)

    model_wrapper = DepthLabWrapper(hyper_param.denoise_steps,
                                    hyper_param.processing_res,
                                    hyper_param.seed,
                                    hyper_param.pretrained_model_name_or_path,
                                    hyper_param.image_encoder_path,
                                    hyper_param.mapping_path,
                                    hyper_param.reference_unet_path,
                                    hyper_param.denoising_unet_path,
                                    hyper_param.normalize_scale,
                                    hyper_param.strength,
                                    hyper_param.blend)

    tester = DepthLabTester(args.dumped_path, 
                            sensor_calib, 
                            hyper_param, 
                            model_wrapper, 
                            args.output_path,
                            args.refine)

    tester.eval()






