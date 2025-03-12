#!/usr/bin/env bash
set -e
# set -x

export CUDA_VISIBLE_DEVICES=2
cd ..

python tests/eval_perf.py \
    --dumped_path '/mnt/nas/perception/yike/validation/dumped_file_v20.1.0/part_2' \
    --process_mode 'inference' \
    --sensor_config '/mnt/nas/perception/datasets/Fusioncart_dataset/v20.1.0/calib/calib.json' \
    --model_hyper_param_config '/home/wangyike/Workspace/playground/depth_completion/DepthLab/tests/configs/model_hyper_param_config.json' \
    --output_path '/mnt/nas/perception/yike/validation/dumped_file_v20.1.0/part_2' \
    --max_range 10 \
    --num_multi_process 1
