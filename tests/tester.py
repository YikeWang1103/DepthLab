import sys
import os
import argparse
import json
import numpy as np
import cv2
import open3d as o3d
import copy
import matplotlib.pyplot as plt

# from matplotlib.colors import LinearSegmentedColormap
from glob import glob
from tqdm import tqdm

from utils.image_util import *


# 定义颜色
# err_colors = ['blue', 'white', 'red']
# err_cmap = LinearSegmentedColormap.from_list('depth_error', err_colors, N=256)
err_cmap = plt.get_cmap('bwr')
depth_cmap = plt.get_cmap('viridis')
mask_cmap = plt.get_cmap('cubehelix')

def depth2colormap(depth_image, max_value, min_value, cmap):
    # 归一化深度图像
    normalized_depth_image = copy.deepcopy(depth_image)
    normalized_depth_image = (normalized_depth_image - min_value) / (max_value - min_value)
    normalized_depth_image = np.clip(normalized_depth_image, 0, 1)

    color_map = cmap(normalized_depth_image)[:, :, :3]  # 丢弃alpha通道
    color_map = (color_map * 255).astype(np.uint8)
    return color_map


def colormap2rgb(rgb_image, colormap, projection_pixels):
    # conver depth_image to the color image
    projection_rgb = copy.deepcopy(rgb_image)

    for pixel in projection_pixels:
        projection_rgb[int(pixel[0]), int(pixel[1])] = colormap[int(pixel[0]), int(pixel[1])]

    return projection_rgb


def generate_mask_from_gt(gt_depth, gt_pixels):
    gt_mask = np.ones(gt_depth.shape)
    
    for gt_pixel in gt_pixels:
        gt_mask[int(gt_pixel[0]), int(gt_pixel[1])] = 0

    return gt_mask


def mask2colormap(mask, cmap):
    mask_colormap = cmap(mask)[:, :, :3]  # 丢弃alpha通道
    mask_colormap = (mask_colormap * 255).astype(np.uint8)
    return mask_colormap


def calculate_rel_err_in_depth(gt_depth, pred_depth, gt_pixels):
    if type(gt_pixels) == list:
        gt_pixels = np.array(gt_pixels)

    rel_err_map = np.zeros([gt_depth.shape[0], gt_depth.shape[1]])
    rel_err_pixels = np.zeros([gt_pixels.shape[0], 3])
    rel_err_pixels[:, :2] = gt_pixels[:, :2]

    for i in range(gt_pixels.shape[0]):
        pred_value = pred_depth[int(gt_pixels[i,0]), int(gt_pixels[i,1])]
        gt_value = gt_depth[int(gt_pixels[i,0]), int(gt_pixels[i,1])]
        rel_err_pixels[i,2] = (pred_value - gt_value) / gt_value
        rel_err_map[int(rel_err_pixels[i,0]), int(rel_err_pixels[i,1])] = rel_err_pixels[i,2]

    return rel_err_map, rel_err_pixels


class DepthLabTester:
    def __init__(self, dumped_path, sensor_calib, hyper_param, model_wrapper, output_path, refine, use_depth_mask):
        self.dumped_path = dumped_path
        self.sensor_calib = sensor_calib
        self.hyper_param = hyper_param
        self.model_wrapper = model_wrapper
        self.output_path = output_path
        self.refine = refine
        self.use_depth_mask = use_depth_mask

        self.visual_folder = "visualization"
        self.pred_folder = "prediction"
        self.hyper_list = str(hyper_param.denoise_steps) +  \
                          "_" + str(hyper_param.processing_res) + \
                          "_" + str(hyper_param.normalize_scale) + \
                          "_" + str(hyper_param.strength) + \
                          "_" + str(hyper_param.blend) + \
                          "_" + str(refine) + \
                          "_" + str(use_depth_mask)

        # output_folder = "inference_results"
        # self.output_folder_path = os.path.join(self.output_path, output_folder)
        # if not os.path.exists(self.output_folder_path):
        #     os.makedirs(self.output_folder_path)

    def eval(self):
        # make sure the dumped path exists, otherwise raise an error
        assert os.path.exists(self.dumped_path), f"{self.dumped_path} does not exist!"

        # for each sequence in the dumped path, load the files in each sequence and evaluate the model
        sequences = os.listdir(self.dumped_path)
        sequences.sort()

        print(f"sequences: {sequences}")
        
        seqs_bar = tqdm(sequences, desc="Sequences", leave=False)

        for seq in seqs_bar:
            seq_path = os.path.join(self.dumped_path, seq)

            seq_path_gt_depth = os.path.join(seq_path, "gt_depth")
            seq_path_pred_depth = os.path.join(seq_path, "pred_depth")
            seq_path_input_depth = os.path.join(seq_path, "input_depth")
            seq_path_input_rgb = os.path.join(seq_path, "input_rgb_0")
            seq_path_depth_mask = os.path.join(seq_path, "sky_mask_in_depth/mask")

            # create the output folder for the sequence
            output_folder_path = os.path.join(seq_path, 'depth_completion', self.hyper_list)
            os.makedirs(output_folder_path, exist_ok=True)

            visual_folder_path = os.path.join(output_folder_path, self.visual_folder)
            os.makedirs(visual_folder_path, exist_ok=True)

            pred_folder_path = os.path.join(output_folder_path, self.pred_folder)
            os.makedirs(pred_folder_path, exist_ok=True)

            gt_depth_files = glob(os.path.join(seq_path_gt_depth, "*.{}".format("bin")))
            input_depth_files = glob(os.path.join(seq_path_input_depth, "*.{}".format("png")))
            input_rgb_files = glob(os.path.join(seq_path_input_rgb, "*.{}".format("png")))
            depth_mask_files = glob(os.path.join(seq_path_depth_mask, "*.{}".format("npy"))) 

            gt_depth_files.sort()
            input_depth_files.sort()
            input_rgb_files.sort()
            depth_mask_files.sort()

            files_bar = tqdm(zip(gt_depth_files, input_rgb_files, depth_mask_files), 
                             total=len(gt_depth_files),
                             desc=" " * 2 + "Frames", 
                             leave=False)

            for gt_depth_file, input_rgb_file, depth_mask_file in files_bar:
                # load input data
                input_rgb = Image.open(input_rgb_file)
                rgb_image = cv2.imread(input_rgb_file)
                gt_depth = np.fromfile(gt_depth_file, dtype=np.float32).reshape(int(self.sensor_calib.RESOLUTION[1]), int(self.sensor_calib.RESOLUTION[0]))
                depth_mask = np.load(depth_mask_file)

                # convert gt depth into colormap
                gt_colormap = depth2colormap(gt_depth, 10, 0.1, depth_cmap)

                # project lidar points in rgb image
                rows, cols = np.where(gt_depth!=0)
                gt_pixels = list(zip(rows, cols))
                projection_rgb = colormap2rgb(rgb_image, gt_colormap, gt_pixels)

                # generate mask for lidar gt
                gt_mask = generate_mask_from_gt(gt_depth, gt_pixels)
                gt_mask_colormap = mask2colormap(gt_mask, mask_cmap)

                # fill the sparse depth map by interpolation
                if self.refine is False:
                    filled_gt_depth=get_filled_for_latents(gt_mask, gt_depth)


                # set the sky depth in the filled_gt_depth to be 10
                if self.use_depth_mask:
                    filled_gt_depth[depth_mask==1] = 10

                filled_gt_depth_colormap = depth2colormap(filled_gt_depth, 10, 0.1, depth_cmap)

                # model inference
                result = self.model_wrapper.run_inference(input_rgb, gt_mask, filled_gt_depth)

                # convert predicted depth into colormap
                depth_pred: np.ndarray = result.depth_np
                depth_pred_colormap = depth2colormap(depth_pred, 10, 0.1, depth_cmap)

                # calculate relative error in depth
                rel_err_arr, rel_err_pixels = calculate_rel_err_in_depth(gt_depth, depth_pred, gt_pixels)
                rel_err_colormap = depth2colormap(rel_err_arr, 0.2, -0.2, err_cmap)
                rel_err_projection_rgb = colormap2rgb(rgb_image, rel_err_colormap, gt_pixels)

                # stitch and save images
                stitched_image = cv2.vconcat([cv2.hconcat([rgb_image, projection_rgb, gt_mask_colormap]),
                                              cv2.hconcat([filled_gt_depth_colormap, depth_pred_colormap, rel_err_projection_rgb])])
                
                cv2.imwrite(os.path.join(visual_folder_path, input_rgb_file.split('/')[-1]), stitched_image)
                
                # save the prediction result as .bin
                depth_pred.tofile(os.path.join(pred_folder_path, input_rgb_file.split('/')[-1].replace('.png', '.bin')))

                pred_check = np.fromfile(os.path.join(pred_folder_path, input_rgb_file.split('/')[-1].replace('.png', '.bin')))

                # print(f"max. value in depth_pred: {np.max(depth_pred)}")
                # print(f"max.value in gt_depth: {np.max(gt_depth)}")


