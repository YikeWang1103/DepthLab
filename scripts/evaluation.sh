#!/usr/bin/env bash
set -e
# set -x

export CUDA_VISIBLE_DEVICES=2
cd ..

python tests/eval_perf.py \
    --dumped_path '/mnt/nas/perception/yike/validation/temp' \
    --sensor_config '/mnt/nas/perception/datasets/Fusioncart_dataset/v19.0.0/calib/calib.json' \
    --model_hyper_param_config '/home/wangyike/Workspace/playground/depth_completion/DepthLab/tests/configs/model_hyper_param_config.json' \
    --output_path '/mnt/nas/perception/yike/validation/temp'