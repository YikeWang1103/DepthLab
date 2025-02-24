import sys
import os
import argparse
import json
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import copy

# 定义颜色
colors = ['blue', 'white', 'red']

# 创建自定义 colormap
cmap = LinearSegmentedColormap.from_list('depth_error', colors, N=256)

# 获取当前工作目录
current_dir = os.getcwd()

# 将当前工作目录添加到 sys.path
if current_dir not in sys.path:
    sys.path.append(current_dir)

from depthlab_wrapper import *
    
def arg_parse():
    parser = argparse.ArgumentParser(
        description="Run validation based on single-frame depth estimation!"
    )

    parser.add_argument(
        '--data_folder', type=str, required=True
    )

    parser.add_argument(
        '--sensor_calib_path', type=str, required=True
    )

    parser.add_argument(
        '--output_folder', type=str, required=True
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )

    parser.add_argument(
        "--normalize_scale",
        type=float,
        default=1,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=1234, 
        help="Random seed."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='./checkpoints/marigold-depth-v1-0',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default='./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--denoising_unet_path", 
        type=str, 
        default='./checkpoints/DepthLab/denoising_unet.pth',
        required=True, 
        help="Path to depth inpainting model."
    )

    parser.add_argument(
        "--reference_unet_path", 
        type=str, 
        default='./checkpoints/DepthLab/reference_unet.pth',
        required=True, 
        help="Path to depth inpainting model."
    )

    parser.add_argument(
        "--mapping_path", 
        type=str, 
        default='./checkpoints/DepthLab/mapping_layer.pth',
        required=True, 
        help="Path to depth inpainting model."
    )

    parser.add_argument(
        "--blend",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--refine",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
 
    args = parser.parse_args()
    return args


def image_preprocess(img, camera_matrix, dist_coeff, new_camera_matrix, scale_factor):
    # img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    img = cv2.fisheye.undistortImage(img, 
                                    camera_matrix, 
                                    dist_coeff, 
                                    np.eye(3, 3), 
                                    new_camera_matrix)

    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    return img


# get point cloud array in .pcd file
def read_pcd_from_file(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    valid_points = pcd.remove_non_finite_points()

    return valid_points


def lidar_pcd_filtering(point_cloud, extrinsics):
    point_cloud_orig = copy.deepcopy(point_cloud)

    # Move to camera coordinate
    point_cloud = point_cloud.transform(extrinsics)

    # hidden points removal
    # print(f"Original point number: {len(point_cloud_orig.points)}")
    _, pt_map = point_cloud.hidden_point_removal([0,0,0], 10000)
    filtered_points = point_cloud_orig.select_by_index(pt_map)
    # print(f"Filtered point number: {len(filtered_points.points)}")

    filtered_points = np.asarray(filtered_points.points)

    return filtered_points


def project_points_to_pixels(camera_matrix, camera_extrinsics, point_cloud):
    """
    Project 3D points to 2D pixel coordinates with depth.
    
    Args:
        camera_matrix: (3, 3) Camera intrinsic matrix
        camera_extrinsics: (4, 4) Camera extrinsic matrix
        point_cloud: (4, N) Homogeneous 3D points
        
    Returns:
        pixel_coords: (3, N) Array of [u, v, depth] for each point
    """
    # Transform points to camera space and project to image plane
    camera_matrix_ = camera_matrix
    camera_matrix = np.eye(4)
    camera_matrix[:3,:3] = camera_matrix_

    pixel_coords = camera_matrix @ camera_extrinsics @ point_cloud
    
    pixel_coords = pixel_coords[:3, :]  # Keep depth for division
    pixel_coords[:2] = pixel_coords[:2] / pixel_coords[2, :]  # Normalize by depth

    return pixel_coords


def pcd2depth(point_cloud, camera_matrix, camera_extrinsics, image_width, image_height, max_range, min_range):
    depth_map = np.zeros((image_height, image_width), dtype=np.float32)

    # transfrom point_cloud from lidar coordinate to camera coordinate
    homo_point_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1)))).transpose()
 
    pixel_coords = project_points_to_pixels(camera_matrix, camera_extrinsics, homo_point_cloud)

    min_x, max_x = 0, image_width
    min_y, max_y = 0, image_height
    x_mask = (pixel_coords[0, :] >= min_x) & (pixel_coords[0, :] < max_x)
    y_mask = (pixel_coords[1, :] >= min_y) & (pixel_coords[1, :] < max_y)
    z_mask = (pixel_coords[2, :] >= min_range) & (pixel_coords[2, :] < max_range)
    mask = x_mask & y_mask & z_mask

    pixel_coords = pixel_coords[:, mask].transpose()

    for pixel_coord in pixel_coords:
        depth_map[int(pixel_coord[1]), int(pixel_coord[0])] = pixel_coord[2]

    return depth_map, pixel_coords


def depth2colormap(depth_image, max_value, min_value):
    normalized_depth_image = (depth_image - min_value) / (max_value - min_value)
    normalized_depth_image = np.clip(normalized_depth_image, 0, 1)

    # 使用Matplotlib的colormap将归一化深度值转换为颜色
    map = plt.get_cmap('jet')
    color_map = map(normalized_depth_image)[:, :, :3]  # 丢弃alpha通道
    color_map = (color_map * 255).astype(np.uint8)

    return color_map


def deptherr2colormap(depth_err_image, max_value, min_value):
    normalized_depth_err_image = (depth_err_image - min_value) / (max_value - min_value)
    normalized_depth_err_image = np.clip(normalized_depth_err_image, 0, 1)

    # 使用Matplotlib的colormap将归一化深度值转换为颜色
    color_map = cmap(normalized_depth_err_image)[:, :, :3]  # 丢弃alpha通道
    color_map = (color_map * 255).astype(np.uint8)

    return color_map


def colormap2rgb(rgb_image, color_map, depth_in_pixel):
    # conver depth_image to the color image
    projection_rgb = copy.deepcopy(rgb_image)

    for pixel in depth_in_pixel:
        projection_rgb[int(pixel[1]), int(pixel[0])] = color_map[int(pixel[1]), int(pixel[0])]

    return projection_rgb


def depth2mask(depth_image):
    depth_mask = np.zeros(depth_image.shape)
    depth_mask[depth_image==0] = 1

    return depth_mask


def calculate_rel_err_in_depth(gt_depth, pred_depth, gt_pixels):
    # gt_depth = gt_depth.astype(np.float32)
    # pred_depth = pred_depth.astype(np.float32)

    rel_err_map = np.zeros([gt_depth.shape[0], gt_depth.shape[1]])
    rel_err_pixels = copy.deepcopy(gt_pixels)

    for i in range(gt_pixels.shape[0]):
        pred_value = pred_depth[int(gt_pixels[i,1]), int(gt_pixels[i,0])]
        gt_value = gt_depth[int(gt_pixels[i,1]), int(gt_pixels[i,0])]
        rel_err_pixels[i,2] = (pred_value - gt_value) / gt_value
        rel_err_map[int(rel_err_pixels[i,1]), int(rel_err_pixels[i,0])] = rel_err_pixels[i,2]

    return rel_err_map, rel_err_pixels


def calculate_abs_err_in_depth(gt_depth, pred_depth, gt_pixels):
    # gt_depth = gt_depth.astype(np.float32)
    # pred_depth = pred_depth.astype(np.float32)

    abs_err_pixels = copy.deepcopy(gt_pixels)

    for i in range(gt_pixels.shape[0]):
        pred_value = pred_depth[gt_depth[i,1], gt_depth[i,0]]
        gt_value = gt_depth [gt_depth[i,1], gt_depth[i,0]]
        abs_err_pixels[i,2] = (pred_value - gt_value)

    return abs_err_pixels


class DepthLabTester:
    def __init__(self, data_folder, sensor_calib_path, output_folder, model_wrapper, refine=False):
        self.imgs = []
        self.pcs = []

        self.data_folder = data_folder
        self.load_data()

        self.sensor_calib_path = sensor_calib_path
        self.load_sensor_calib()

        self.output_folder = output_folder
        self.model_wrapper = model_wrapper
        self.refine = refine

        self.print_sensor_calib()
        self.print_data_info()

    def load_sensor_calib(self):
        with open(self.sensor_calib_path, 'r') as f:
            sensor_calib = json.load(f)

        camera_calib = sensor_calib['mono camera']
        self.camera_width = camera_calib['resolution'][0]
        self.camera_height = camera_calib['resolution'][1]

        self.scale_factor = 0.5

        intrinsics = camera_calib['intrinsics']
        self.camera_matrix = np.array(intrinsics['k'])
        self.camera_dist = np.array(intrinsics['dist'])

        self.camera_extrinsic = np.array(camera_calib['lidar2camera'])

        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix,
                                                                    self.camera_dist,
                                                                    (self.camera_width, self.camera_height),
                                                                    0,
                                                                    (self.camera_width, self.camera_height))
        
        self.scaled_new_camera_matrix = copy.deepcopy(self.new_camera_matrix)
        self.scaled_new_camera_matrix[0:2, :] = self.new_camera_matrix[0:2, :] * self.scale_factor

    def load_data(self):
        for folder in os.listdir(self.data_folder):
            folder_path = os.path.join(self.data_folder, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.png'):
                        img_path = os.path.join(folder_path, file)
                        self.imgs.append(img_path)
                    elif file.endswith('.pcd'):
                        pc_path = os.path.join(folder_path, file)
                        self.pcs.append(pc_path)
                    else:
                        assert False, f"Unknown file type: {file}"

        if len(self.imgs) != len(self.pcs):
            assert False, f"imgs and pcs have different length"

        self.imgs = sorted(self.imgs)
        self.pcs = sorted(self.pcs)

    def print_sensor_calib(self):
        print(f"Camera Calibration Information:")
        print(f"sensor_calib_path: {self.sensor_calib_path}")
        print(f"camera_matrix: {self.camera_matrix}")
        print(f"new_camera_matrix: {self.new_camera_matrix}")
        print(f"scaled_new_camera_matrix: {self.scaled_new_camera_matrix}")
        print(f"camera_dist: {self.camera_dist}")
        print(f"camera_extrinsic: {self.camera_extrinsic}")

    def print_data_info(self):
        print(f"Data Information:")
        print(f"data_folder: {self.data_folder}")
        print(f"imgs: {self.imgs}")
        print(f"pcs: {self.pcs}")


    def save_intermedia_results(self, 
                                output_folder_path,
                                depth_image,
                                filled_depth_color_map,
                                input_image,
                                color_map,
                                projection_rgb,
                                depth_mask,
                                rel_err_projection_rgb):
        
        # save the depth image
        depth_image_path = os.path.join(output_folder_path, 'depth.png')
        cv2.imwrite(depth_image_path, depth_image)

        # save the filled depth colormap
        filled_depth_color_map_path = os.path.join(output_folder_path, 'filled_depth_color_map.png')
        cv2.imwrite(filled_depth_color_map_path, filled_depth_color_map)

        # save depth_mask as .npy
        depth_mask_path = os.path.join(output_folder_path, 'depth_mask.npy')
        np.save(depth_mask_path, depth_mask)

        # save the input image
        input_image_path = os.path.join(output_folder_path, 'input.png')
        cv2.imwrite(input_image_path, input_image)

        # save the depth image in color
        depth_image_in_color = cv2.applyColorMap(color_map, cv2.COLORMAP_JET)
        depth_image_in_color_path = os.path.join(output_folder_path, 'depth_in_color.png')
        cv2.imwrite(depth_image_in_color_path, depth_image_in_color)

        # save the projection rgb image
        projection_rgb_path = os.path.join(output_folder_path, 'projection_rgb.png')
        cv2.imwrite(projection_rgb_path, projection_rgb)

        # save the depth mask image
        depth_mask_color_image = (depth_mask * 255).astype(np.uint8)
        depth_mask_in_color = cv2.applyColorMap(depth_mask_color_image, cv2.COLORMAP_JET)
        depth_mask_path = os.path.join(output_folder_path, 'depth_mask.png')
        cv2.imwrite(depth_mask_path, depth_mask_in_color)

        # save the relative error projection rgb image
        relative_error_projection_rgb_path = os.path.join(output_folder_path, 'relative_error_projection_rgb.png')
        cv2.imwrite(relative_error_projection_rgb_path, rel_err_projection_rgb)

    def run(self):
        for img_path, pcd_path in zip(self.imgs, self.pcs):
            # check and generate output folder path
            sub_folder = img_path.split('/')[-2]
            output_folder_path = os.path.join(self.output_folder, sub_folder)

            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # read, undistort and resize the original image
            img = cv2.imread(img_path)
            rgb_image = image_preprocess(img, self.camera_matrix, self.camera_dist, self.new_camera_matrix, self.scale_factor)

            # read pcd file and filter the point cloud
            valid_point_cloud = read_pcd_from_file(pcd_path)
            filtered_point_cloud = lidar_pcd_filtering(valid_point_cloud, self.camera_extrinsic)

            # project point cloud to the depth image
            depth_image, depth_in_pixel = pcd2depth(filtered_point_cloud, 
                                                    self.scaled_new_camera_matrix, 
                                                    self.camera_extrinsic, 
                                                    int(self.camera_width * self.scale_factor), 
                                                    int(self.camera_height * self.scale_factor),
                                                    100,
                                                    0)

            # convert the depth image to depth mask
            depth_mask = depth2mask(depth_image)

            # fill the sparse depth map by interpolation
            if self.refine is False:
                filled_depth_image=get_filled_for_latents(depth_mask, depth_image)
            
            input_image = Image.open(os.path.join(output_folder_path, 'input.png'))

            result = self.model_wrapper.run_inference(input_image, depth_mask, filled_depth_image)

            depth_pred: np.ndarray = result.depth_np

            # calculate the depth error map
            rel_err_depthmap, rel_err_depth_in_pixel = calculate_rel_err_in_depth(depth_image, depth_pred, depth_in_pixel)
            rel_err_colormap = deptherr2colormap(rel_err_depthmap, 0.2, -0.2)
            rel_err_projection_rgb = colormap2rgb(rgb_image, rel_err_colormap, rel_err_depth_in_pixel)

            # np.save(npy_save_path, depth_pred)

            # save the prediction results for visualization
            result.depth_colored.save(os.path.join(output_folder_path, 'pred.png'))

            # generate lidar point cloud projection on rgb image
            min_depth = np.max([np.min(depth_image), 0])
            max_depth = np.min([np.max(depth_image),100])
            depth_color_map  = depth2colormap(depth_image, max_depth, min_depth)
            projection_rgb = colormap2rgb(rgb_image, depth_color_map, depth_in_pixel)
            filled_depth_color_map = depth2colormap(filled_depth_image, max_depth, min_depth)

            # save intermedia results for visualization
            self.save_intermedia_results(output_folder_path,
                                         depth_image,
                                         filled_depth_color_map,
                                         rgb_image,
                                         depth_color_map,
                                         projection_rgb,
                                         depth_mask,
                                         rel_err_projection_rgb)


if __name__ == '__main__':
    args = arg_parse()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    model_wrapper = DepthLabWrapper(args.denoise_steps,
                                    args.processing_res,
                                    args.seed,
                                    args.pretrained_model_name_or_path,
                                    args.image_encoder_path,
                                    args.mapping_path,
                                    args.reference_unet_path,
                                    args.denoising_unet_path,
                                    args.normalize_scale,
                                    args.strength,
                                    args.blend)

    tester = DepthLabTester(args.data_folder, args.sensor_calib_path, args.output_folder, model_wrapper, args.refine)
    tester.run()




