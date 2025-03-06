#!/usr/bin/env bash
set -e
# set -x

export CUDA_VISIBLE_DEVICES=0
cd ..

python tests/eval_perf.py \
    --dumped_path '/mnt/nas/perception/yike/validation/dumped_file_val_v19.1.0' \
    --sensor_config '/mnt/nas/perception/datasets/Fusioncart_dataset/v19.1.0/calib/calib.json' \
    --model_hyper_param_config '/home/wangyike/Workspace/playground/depth_completion/DepthLab/tests/configs/model_hyper_param_config.json' \
    --output_path '/mnt/nas/perception/yike/validation/dumped_file_val_v19.1.0' \
    --use_depth_mask