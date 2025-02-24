import argparse
import logging
import os
import random
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
import cv2
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_2d_condition_main import UNet2DConditionModel_main
from src.models.projection import My_proj
from transformers import CLIPVisionModelWithProjection
from inference.depthlab_pipeline import DepthLabPipeline
from utils.seed_all import seed_all
from utils.image_util import get_filled_for_latents


class DepthLabWrapper:
    def __init__(self, 
                 denoise_steps, 
                 processing_res, 
                 seed,
                 pretrained_model_name_or_path,
                 image_encoder_path,
                 mapping_path,
                 reference_unet_path,
                 denoising_unet_path,
                 normalize_scale,
                 strength,
                 blend):
        
        self.denoise_steps = denoise_steps
        self.processing_res = processing_res
        self.normalize_scale = normalize_scale
        self.strength = strength
        self.blend = blend

        if seed is None:
            import time
            self.seed = int(time.time())

        seed_all(seed)

        # -------------------- Device --------------------
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
        logging.info(f"device = {self.device}") 

        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, 
                                                 subfolder='vae')

        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, 
                                                          subfolder='text_encoder')

        self.denoising_unet = UNet2DConditionModel_main.from_pretrained(pretrained_model_name_or_path,
                                                                        subfolder="unet",
                                                                        in_channels=12, 
                                                                        sample_size=96,
                                                                        low_cpu_mem_usage=False,
                                                                        ignore_mismatched_sizes=True)
        
        self.reference_unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path,
                                                                   subfolder="unet",
                                                                   in_channels=4, 
                                                                   sample_size=96,
                                                                   low_cpu_mem_usage=False,
                                                                   ignore_mismatched_sizes=True)
        
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)

        self.mapping_layer=My_proj()

        self.mapping_layer.load_state_dict(torch.load(mapping_path, map_location="cpu"), strict=False,)
        
        self.mapping_device = torch.device("cuda")

        self.mapping_layer.to(self.mapping_device)

        self.reference_unet.load_state_dict(torch.load(reference_unet_path, map_location="cpu"),)

        self.denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False,)

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer')

        self.scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder='scheduler')

        self.pipe = DepthLabPipeline(reference_unet=self.reference_unet,
                                     denoising_unet=self.denoising_unet,  
                                     mapping_layer=self.mapping_layer,
                                     vae=self.vae,
                                     text_encoder=self.text_encoder,
                                     tokenizer=self.tokenizer,
                                     image_enc=self.image_enc,
                                     scheduler=self.scheduler,).to('cuda')
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            logging.debug("run without xformers")

    def run_inference(self, input_image, mask, depth_numpy):
        pipe_out = self.pipe(input_image,
                             denosing_steps = self.denoise_steps,
                             processing_res = self.processing_res,
                             match_input_res = True,
                             batch_size =1,
                             color_map = "Spectral",
                             show_progress_bar = False,
                             depth_numpy_origin = depth_numpy,
                             mask_origin = mask,
                             guidance_scale = 1,
                             normalize_scale = self.normalize_scale,
                             strength = self.strength,
                             blend = self.blend)
        
        return pipe_out

