
#!/usr/bin/env bash
set -e
# set -x

pretrained_model_name_or_path='./checkpoints/marigold-depth-v1-0'
image_encoder_path='./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K'
denoising_unet_path='./checkpoints/DepthLab/denoising_unet.pth'
reference_unet_path='./checkpoints/DepthLab/reference_unet.pth'
mapping_path='./checkpoints/DepthLab/mapping_layer.pth'

export CUDA_VISIBLE_DEVICES=0
cd ..

python tests/test.py  \
    --data_folder 'test_cases/yard' \
    --sensor_calib_path 'tests/configs/sc230ai_binocular_camera_calibration_2024_12_16.json' \
    --output_folder 'output/yard' \
    --seed 1234 \
    --denoise_steps 20 \
    --processing_res 640 \
    --normalize_scale 1 \
    --strength 0.8 \
    --pretrained_model_name_or_path './checkpoints/marigold-depth-v1-0' \
    --image_encoder_path './checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K' \
    --denoising_unet_path './checkpoints/DepthLab/denoising_unet.pth' \
    --reference_unet_path './checkpoints/DepthLab/reference_unet.pth' \
    --mapping_path './checkpoints/DepthLab/mapping_layer.pth' \
    --blend 