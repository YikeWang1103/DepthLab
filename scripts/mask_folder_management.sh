#!/bin/bash

# 设置基础路径
BASE_PATH="/mnt/nas/perception/datasets/Fusioncart_dataset/v19.1.0"

# 遍历基础路径下的所有文件夹
for dir in "$BASE_PATH"/*/ ; do
    if [ -d "$dir" ]; then
        echo "处理文件夹: $dir"
        
        # # 检查masks/depth_mask是否存在并重命名
        # if [ -d "${dir}completion_depth/50_640_1.0_0.8_True_False_False" ]; then
        #     echo "${dir}completion_depth/50_640_1.0_0.8_True_False_False exists!"
        #     mv "${dir}completion_depth/50_640_1.0_0.8_True_False_False" "${dir}completion_depth/50_640_1.0_0.8_True_False_False_False"
        # fi

        # 检查文件数量是否一致
        INPUT_PATH="${dir}input_rgb_0"
        PREDICTION_PATH="${dir}completion_depth/50_640_1.0_0.8_True_False_False_False/prediction"

        if [ -d "$INPUT_PATH" ] && [ -d "$PREDICTION_PATH" ]; then
            INPUT_COUNT=$(find "$INPUT_PATH" -type f | wc -l)
            PREDICTION_COUNT=$(find "$PREDICTION_PATH" -type f | wc -l)

            if [ "$INPUT_COUNT" -eq "$PREDICTION_COUNT" ]; then
                # echo "文件数量一致: $INPUT_COUNT"
                continue
            else
                echo "文件数量不一致: 输入文件数量 $INPUT_COUNT, 预测文件数量 $PREDICTION_COUNT"
            fi
        fi

    fi
done

echo "所有文件夹处理完成！"