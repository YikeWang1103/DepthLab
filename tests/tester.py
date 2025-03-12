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

def add_mask_overlay(rgb_image, mask, color=(0, 255, 0), alpha=0.4):
    """
    在RGB图像上添加半透明的mask效果
    
    参数:
        rgb_image: RGB格式的图像数组
        mask: 二值mask图像 (0或1)
        color: mask的颜色，默认为绿色 (B,G,R)
        alpha: 透明度，0-1之间，1为完全不透明
    """
    # 确保mask是二值图像
    mask = mask.astype(bool)
    
    # 创建彩色mask
    colored_mask = np.zeros_like(rgb_image)
    colored_mask[mask] = color
    
    # 将mask叠加到原图上
    overlay = cv2.addWeighted(rgb_image, alpha, colored_mask, 1, 0)
    
    return overlay



class DepthLabTester:
    def __init__(self, dumped_path, sensor_calib, hyper_param, 
                 model_wrapper, output_path, max_range, 
                 refine, use_geometric_mask, use_semantic_mask):
        
        self.dumped_path = dumped_path
        self.sensor_calib = sensor_calib
        self.hyper_param = hyper_param
        self.model_wrapper = model_wrapper
        self.output_path = output_path
        self.max_range = max_range
        self.refine = refine
        self.use_geometric_mask = use_geometric_mask
        self.use_semantic_mask = use_semantic_mask

        self.pred_folder = "prediction"
        self.visual_folder = "prediction_visualization"
        
        self.hyper_list = str(hyper_param.denoise_steps) +  \
                          "_" + str(hyper_param.processing_res) + \
                          "_" + str(hyper_param.normalize_scale) + \
                          "_" + str(hyper_param.strength) + \
                          "_" + str(hyper_param.blend) + \
                          "_" + str(refine) + \
                          "_" + str(use_geometric_mask) + \
                          "_" + str(use_semantic_mask)

        # output_folder = "inference_results"
        # self.output_folder_path = os.path.join(self.output_path, output_folder)
        # if not os.path.exists(self.output_folder_path):
        #     os.makedirs(self.output_folder_path)

    def run(self):
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
            seq_path_input_depth = os.path.join(seq_path, "input_depth")
            seq_path_input_rgb = os.path.join(seq_path, "input_rgb_0")
            seq_path_depth_mask = os.path.join(seq_path, "masks/geometric_masks/masks")

            # create the output folder for the sequence
            output_folder_path = os.path.join(seq_path, 'completion_depth', self.hyper_list)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path,)

            visual_folder_path = os.path.join(output_folder_path, self.visual_folder)
            if not os.path.exists(visual_folder_path):
                os.makedirs(visual_folder_path)

            pred_folder_path = os.path.join(output_folder_path, self.pred_folder)
            if not os.path.exists(pred_folder_path):
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
                gt_colormap = depth2colormap(gt_depth, self.max_range, 0.1, depth_cmap)

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
                if self.use_geometric_mask:
                    filled_gt_depth[depth_mask==1] = 10

                filled_gt_depth_colormap = depth2colormap(filled_gt_depth, self.max_range, 0.1, depth_cmap)

                # model inference
                result = self.model_wrapper.run_inference(input_rgb, gt_mask, filled_gt_depth)

                # convert predicted depth into colormap
                depth_pred: np.ndarray = result.depth_np
                depth_pred_colormap = depth2colormap(depth_pred, self.max_range, 0.1, depth_cmap)

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

                print(f"max. value in depth_pred: {np.max(depth_pred)}")
                print(f"max.value in gt_depth: {np.max(gt_depth)}")

    def refine_with_sky_mask(self, process_id, cpu_num):
        # make sure the dumped path exists, otherwise raise an error
        assert os.path.exists(self.dumped_path), f"{self.dumped_path} does not exist!"

        # for each sequence in the dumped path, load the files in each sequence and evaluate the model
        sequences = os.listdir(self.dumped_path)
        sequences.sort()

        sub_seqs = sequences[(len(sequences) // cpu_num + 1) * process_id : (len(sequences) // cpu_num + 1) * (process_id + 1)]

        print(f"sub_seqs: {sub_seqs}")
        
        sub_seqs_bar = tqdm(sub_seqs, desc="Sub Sequences", leave=False)

        for seq in sub_seqs_bar:
            seq_path = os.path.join(self.dumped_path, seq)

            seq_path_completion_depth = os.path.join(seq_path, "completion_depth", self.hyper_list)
            seq_path_completion_depth_prediction = os.path.join(seq_path_completion_depth, "prediction")
            seq_path_input_rgb = os.path.join(seq_path, "input_rgb_0")
            seq_path_geometric_mask = os.path.join(seq_path, "masks/geometric_masks/masks")
            seq_path_semantic_mask = os.path.join(seq_path, "masks/semantic_masks/masks")

            processed_depth_path = os.path.join(seq_path_completion_depth, "processed")
            processed_depth_viz_path = os.path.join(seq_path_completion_depth, "processed_visualization")

            geometric_mask_depth_path = os.path.join(processed_depth_path, "geometric")
            semantic_mask_depth_path = os.path.join(processed_depth_path, "semantic")
            geometric_and_semantic_mask_depth_path = os.path.join(processed_depth_path, "geometric_and_semantic")
            geometric_or_semantic_mask_depth_path = os.path.join(processed_depth_path, "geometric_or_semantic")

            
            # 确保文件路径存在，否则抛出错误
            assert os.path.exists(seq_path_completion_depth_prediction), f"{seq_path_completion_depth_prediction} does not exist!"
            assert os.path.exists(seq_path_input_rgb), f"{seq_path_input_rgb} does not exist!"
            assert os.path.exists(seq_path_geometric_mask), f"{seq_path_geometric_mask} does not exist!"
            assert os.path.exists(seq_path_semantic_mask), f"{seq_path_semantic_mask} does not exist!"

            # 如何指定文件夹不存在，创建该文件夹
            if not os.path.exists(processed_depth_path):
                os.makedirs(processed_depth_path)

            if not os.path.exists(processed_depth_viz_path):
                os.makedirs(processed_depth_viz_path)

            if not os.path.exists(geometric_mask_depth_path):
                os.makedirs(geometric_mask_depth_path)

            if not os.path.exists(semantic_mask_depth_path):
                os.makedirs(semantic_mask_depth_path)

            if not os.path.exists(geometric_and_semantic_mask_depth_path):
                os.makedirs(geometric_and_semantic_mask_depth_path)

            if not os.path.exists(geometric_or_semantic_mask_depth_path):
                os.makedirs(geometric_or_semantic_mask_depth_path)


            completion_depth_prediction_files = glob(os.path.join(seq_path_completion_depth_prediction, "*.{}".format("bin")))
            input_rgb_files = glob(os.path.join(seq_path_input_rgb, "*.{}".format("png")))
            geometric_mask_files = glob(os.path.join(seq_path_geometric_mask, "*.{}".format("npy")))
            semantic_mask_files = glob(os.path.join(seq_path_semantic_mask, "*.{}".format("npy")))

            completion_depth_prediction_files.sort()
            input_rgb_files.sort()
            geometric_mask_files.sort()
            semantic_mask_files.sort()

            # print(f"completion_depth_prediction_files.shape: {len(completion_depth_prediction_files)}")
            # print(f"input_rgb_files.shape: {len(input_rgb_files)}")
            # print(f"geometric_mask_files.shape: {len(geometric_mask_files)}")
            # print(f"semantic_mask_files.shape: {len(semantic_mask_files)}")

            files_bar = tqdm(zip(completion_depth_prediction_files, input_rgb_files, geometric_mask_files, semantic_mask_files), 
                                total=len(completion_depth_prediction_files),
                                desc=" " * 2 + "Frames", 
                                leave=False)

            for completion_depth_prediction_file, \
                input_rgb_file, \
                geometric_mask_file, \
                semantic_mask_file in files_bar:

                completion_depth_prediction = np.fromfile(completion_depth_prediction_file).reshape(int(self.sensor_calib.RESOLUTION[1]), int(self.sensor_calib.RESOLUTION[0]))
                rgb_image = cv2.imread(input_rgb_file)
                geometric_mask = np.load(geometric_mask_file)
                semantic_mask = np.load(semantic_mask_file)

                geometric_mask_depth = copy.deepcopy(completion_depth_prediction)
                geometric_mask_depth[geometric_mask==1] = self.max_range

                semantic_mask_depth = copy.deepcopy(completion_depth_prediction)
                semantic_mask_depth[semantic_mask==1] = self.max_range

                geometric_and_semantic_mask_depth = copy.deepcopy(completion_depth_prediction)
                geometric_and_semantic_mask = np.logical_and(geometric_mask==1, semantic_mask==1)
                geometric_and_semantic_mask_depth[geometric_and_semantic_mask] = self.max_range

                geometric_or_semantic_mask_depth = copy.deepcopy(completion_depth_prediction)
                geometric_or_semantic_mask = np.logical_or(geometric_mask==1, semantic_mask==1)
                geometric_or_semantic_mask_depth[geometric_or_semantic_mask] = self.max_range

                # 将处理后的深度图保存为float16格式
                geometric_mask_depth_float16 = geometric_mask_depth.astype(np.float16)
                geometric_mask_depth_filepath = os.path.join(geometric_mask_depth_path, os.path.basename(completion_depth_prediction_file).replace('.bin', '.npy'))
                np.save(geometric_mask_depth_filepath, geometric_mask_depth_float16)

                semantic_mask_depth_float16 = semantic_mask_depth.astype(np.float16)
                semantic_mask_depth_filepath = os.path.join(semantic_mask_depth_path, os.path.basename(completion_depth_prediction_file).replace('.bin', '.npy'))
                np.save(semantic_mask_depth_filepath, semantic_mask_depth_float16)

                geometric_and_semantic_mask_depth_float16 = geometric_and_semantic_mask_depth.astype(np.float16)
                geometric_and_semantic_mask_depth_filepath = os.path.join(geometric_and_semantic_mask_depth_path, os.path.basename(completion_depth_prediction_file).replace('.bin', '.npy'))
                np.save(geometric_and_semantic_mask_depth_filepath, geometric_and_semantic_mask_depth_float16)

                geometric_or_semantic_mask_depth_float16 = geometric_or_semantic_mask_depth.astype(np.float16)
                geometric_or_semantic_mask_depth_filepath = os.path.join(geometric_or_semantic_mask_depth_path, os.path.basename(completion_depth_prediction_file).replace('.bin', '.npy'))
                np.save(geometric_or_semantic_mask_depth_filepath, geometric_or_semantic_mask_depth_float16)

                geometric_mask_overlay = add_mask_overlay(rgb_image, geometric_mask, (0,0,255), 0.5)
                semantic_mask_overlay = add_mask_overlay(rgb_image, semantic_mask, (0,0,255), 0.5)
                geometric_and_semantic_mask_overlay = add_mask_overlay(rgb_image, geometric_and_semantic_mask, (0,0,255), 0.5)
                geometric_or_semantic_mask_overlay = add_mask_overlay(rgb_image, geometric_or_semantic_mask, (0,0,255), 0.5)

                completion_depth_colormap = depth2colormap(completion_depth_prediction, self.max_range, 0.1, depth_cmap)
                geometric_mask_depth_colormap = depth2colormap(geometric_mask_depth, self.max_range, 0.1, depth_cmap)
                semantic_mask_depth_colormap = depth2colormap(semantic_mask_depth, self.max_range, 0.1, depth_cmap)
                geometric_and_semantic_mask_depth_colormap = depth2colormap(geometric_and_semantic_mask_depth, self.max_range, 0.1, depth_cmap)
                geometric_or_semantic_mask_depth_colormap = depth2colormap(geometric_or_semantic_mask_depth, self.max_range, 0.1, depth_cmap)

                stitched_image = cv2.vconcat([cv2.hconcat([rgb_image, 
                                                           geometric_mask_overlay, 
                                                           semantic_mask_overlay,
                                                           geometric_and_semantic_mask_overlay,
                                                           geometric_or_semantic_mask_overlay]),
                                              cv2.hconcat([completion_depth_colormap, 
                                                           geometric_mask_depth_colormap, 
                                                           semantic_mask_depth_colormap, 
                                                           geometric_and_semantic_mask_depth_colormap, 
                                                           geometric_or_semantic_mask_depth_colormap])])
                
                cv2.imwrite(os.path.join(processed_depth_viz_path, os.path.basename(input_rgb_file)), stitched_image)


