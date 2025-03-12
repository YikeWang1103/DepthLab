#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import struct
from pathlib import Path
import shutil
from tqdm import tqdm

def bin_to_numpy(bin_file_path):
    """
    将.bin文件转换为numpy数组
    """

    np_data = np.fromfile(bin_file_path).reshape([360, 640])
    np_data = np_data.astype(np.float16)

    return np_data

def process_folders():
    """
    处理指定路径下的所有文件夹，转换.bin文件为.npy文件
    """
    # 源路径
    source_base_path = "/mnt/nas/perception/yike/validation/dumped_file_v19.1.0_temp"
    # 目标路径
    target_base_path = "/mnt/nas/perception/datasets/Fusioncart_dataset/v19.1.0/completion_depth/CAM_FRONT"
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_base_path, exist_ok=True)
    
    # 获取源路径下的所有文件夹
    folders = [f for f in os.listdir(source_base_path) if os.path.isdir(os.path.join(source_base_path, f))]
    
    print(f"找到 {len(folders)} 个文件夹需要处理")
    
    folders_bar = tqdm(folders, desc="处理文件夹")

    # 遍历每个文件夹
    for folder in folders_bar:
        bin_folder_path = os.path.join(source_base_path, folder, "completion_depth/50_640_1.0_0.8_True_False_False_False/prediction")
        
        # 检查bin文件夹是否存在
        if not os.path.exists(bin_folder_path):
            print(f"警告: 在 {folder} 中未找到指定路径")
            continue
        
        # 获取所有.bin文件
        bin_files = [f for f in os.listdir(bin_folder_path) if f.endswith('.bin')]
        bin_files_bar = tqdm(bin_files, desc="处理.bin文件")

        for bin_file in bin_files_bar:
            bin_file_path = os.path.join(bin_folder_path, bin_file)
            
            # 转换.bin文件为numpy数组
            try:
                np_data = bin_to_numpy(bin_file_path)
                
                # 创建目标文件名（将.bin替换为.npy）
                npy_file_name = os.path.splitext(bin_file)[0] + '.npy'
                npy_file_path = os.path.join(target_base_path, npy_file_name)
                
                # 保存numpy数组为.npy文件
                np.save(npy_file_path, np_data)
                
            except Exception as e:
                print(f"处理文件 {bin_file_path} 时出错: {str(e)}")

if __name__ == "__main__":
    process_folders()
    print("转换完成！") 