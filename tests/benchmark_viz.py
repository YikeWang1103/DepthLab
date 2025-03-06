import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def draw_text_in_image(text, image):
    # 在深度图像上添加文件夹名称文本
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)  # 白色
    font_thickness = 2
    
    # 计算文本大小以便放置在图像的左上角
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = 10  # 左边距
    text_y = 30  # 上边距
    
    # 在深度图像上绘制文本
    cv2.putText(image, text, (text_x, text_y), 
                font, font_scale, font_color, font_thickness)
    
    return image


class BenchmarkVizer:
    def __init__(self, input_path, folder_names, output_path):
        """初始化BenchmarkVizer类

        Args:
            input_path (str): 输入文件夹的根路径
            folder_names (list): 包含所有需要对比的文件夹名称的列表
        """
        self.input_path = input_path
        self.folder_names = folder_names
        self.output_path = output_path

        # 创建输出目录（如果不存在）
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def run(self):
        sequences = [name for name in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path, name))]
            
        # 使用tqdm为序列创建进度条
        sequences_bar = tqdm(sequences, desc="处理序列", leave=True)
        for seq in sequences_bar:
            rgb_folder_path = os.path.join(self.input_path, seq, 'input_rgb_0')

            image_names = os.listdir(rgb_folder_path)
            image_names.sort()

            images_bar = tqdm(image_names, desc="处理图片", leave=True)
            for image_name in images_bar:
                rgb_image = cv2.imread(os.path.join(rgb_folder_path, image_name))
                # stitched_image = cv2.vconcat([rgb_image, rgb_image])
                stitched_image = np.array([])

                for inference_folder in self.folder_names:
                    inference_folder_path = os.path.join(self.input_path, seq, inference_folder)

                    inference_viz_image_name = os.path.join(inference_folder_path, 'visualization', image_name)

                    inference_viz_image = cv2.imread(inference_viz_image_name)

                    height = inference_viz_image.shape[0] // 2
                    width = inference_viz_image.shape[1] // 3

                    depth_color_image = inference_viz_image[height:, width:2*width, :]
                    depth_err_color_image = inference_viz_image[height:, 2*width:, :]

                    draw_text_in_image(inference_folder, depth_color_image)

                    if stitched_image.size == 0:
                        stitched_image = cv2.vconcat([depth_color_image, depth_err_color_image])
                    else:
                        stitched_image = cv2.hconcat([stitched_image, cv2.vconcat([depth_color_image, depth_err_color_image])])

                cv2.imwrite(os.path.join(self.output_path, image_name), stitched_image)

# # 获取指定路径下的所有文件夹名
# folder_path = '/mnt/nas/perception/yike/validation/temp'
# folder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

def arg_parser():

    parser = argparse.ArgumentParser(
        description="可视化不同模型的深度预测结果"
    )

    parser.add_argument(
        '--input_path', type=str, required=True,
        help='输入文件夹的输入路径'
    )

    parser.add_argument(
        '--output_path', type=str, required=True,
        help='输入文件夹的输出路径'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()

    folder_names = ['depth_completion/50_640_1.0_0.8_True_False_True', 
                    'depth_completion/50_640_1.0_0.8_True_False_False']
    
    vizer = BenchmarkVizer(args.input_path, folder_names, args.output_path)

    vizer.run()



