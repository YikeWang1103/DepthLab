#!/bin/bash

# 设置基础路径
BASE_PATH="/mnt/nas/perception/yike/validation/dumped_file_v20.1.0/part_3"

# 遍历基础路径下的所有文件夹
for dir in "$BASE_PATH"/*/ ; do
    if [ -d "$dir" ]; then
        echo "处理文件夹: $dir"
        
        #  检查masks/depth_masks是否存在
        if [ -d "${dir}masks/depth_masks" ]; then
            # 将masks/depth_masks重命名为masks/geometric_masks
            mv "${dir}masks/depth_masks" "${dir}masks/geometric_masks"
        fi

        # # 检查文件数量是否一致
        # INPUT_PATH="${dir}input_rgb_0"
        # PREDICTION_PATH="${dir}completion_depth/50_640_1.0_0.8_True_False_False_False/prediction"

        # if [ -d "$INPUT_PATH" ] && [ -d "$PREDICTION_PATH" ]; then
        #     INPUT_COUNT=$(find "$INPUT_PATH" -type f | wc -l)
        #     PREDICTION_COUNT=$(find "$PREDICTION_PATH" -type f | wc -l)

        #     if [ "$INPUT_COUNT" -eq "$PREDICTION_COUNT" ]; then
        #         # echo "文件数量一致: $INPUT_COUNT"
        #         continue
        #     else
        #         echo "文件数量不一致: 输入文件数量 $INPUT_COUNT, 预测文件数量 $PREDICTION_COUNT"
        #     fi
        # fi

    fi
done

echo "所有文件夹处理完成！"