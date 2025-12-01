# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TT
import torchvision.transforms.functional as TF
from einops import rearrange
from icecream import ic
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm import tqdm
from wan.text2video import (
    FlowDPMSolverMultistepScheduler,
    FlowUniPCMultistepScheduler,
    T5EncoderModel,
    WanT2V,
    WanVAE,
    get_sampling_sigmas,
    retrieve_timesteps,
    shard_model,
)

from ..utils.preprocessor import VaceVideoProcessor
from .modules.model import VaceWanModel

import decord  # isort:skip
from decord import cpu # isort:skip


class WanVace(nn.Module):
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        skip_t5=False,
        enable_skeleton_cross_attn=False,
        enable_audio_cross_attn=False,
        use_gradient_checkpointing=True,
        model_path="pretrained_models/Wan2.1_Encoders/diffusion_pytorch_model.safetensors",
        audio_model_path=None
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        super().__init__()
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)

        # Skip T5 loading when using cached embeddings
        if skip_t5:
            logging.info("Skipping T5 model loading (using cached embeddings)")
            self.text_encoder = None
        else:
            t5_device = torch.device("cpu") if t5_cpu else self.device
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=t5_device,     #cpu offload: device=torch.device("cpu")
                checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
                shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        if enable_skeleton_cross_attn or model_path.endswith("ckpt") or enable_audio_cross_attn:
            logging.info(f"Creating WanModel from {model_path}")
            self.model = VaceWanModel.from_ckpt(
                model_path,
                audio_model_path,
                additional_kwargs={
                    "enable_skeleton_cross_attn": enable_skeleton_cross_attn,
                    "enable_audio_cross_attn": enable_audio_cross_attn,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                },
            )
            self.model.eval().requires_grad_(False)
        else:
            logging.info(f"Creating VaceWanModel from {checkpoint_dir}")
            self.model = VaceWanModel.from_pretrained(checkpoint_dir)
            self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
                usp_dit_forward_vace,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            for block in self.model.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.model.forward_vace = types.MethodType(usp_dit_forward_vace, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def vace_encode_frames(self, frames, ref_images, masks=None):
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = self.vae.encode(frames)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]     # frames:[[c f h w]]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = self.vae.encode(inactive)
            reactive = self.vae.encode(reactive)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        ref_latent_for_noise = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = self.vae.encode(refs)
                else:
                    ref_latent = self.vae.encode(refs)
                    ref_latent_for_noise.append(ref_latent[0])
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents, ref_latent_for_noise

    def vace_encode_masks(self, masks, ref_images=None):
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // self.vae_stride[0])
            height = 2 * (int(height) // (self.vae_stride[1] * 2))
            width = 2 * (int(width) // (self.vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, self.vae_stride[1], width, self.vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                self.vae_stride[1] * self.vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames, image_size, device):
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_video is None:
                src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i] = self.load_video(sub_src_video)#c f h w
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones((3,src_video[0].shape[1],image_size[0],image_size[1]), device=device)
                image_sizes.append(src_video[i].shape[2:])
        
        target_shape = image_size
        ori_image_sizes=[]
        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = Image.open(ref_img).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        ori_image_sizes.append(ref_img.shape[-2:])
                        ref_img = self._resize_for_rectangle_crop(ref_img, target_shape, reshape_mode="center")
                        # if ref_img.shape[-2:] != image_size:
                        #     canvas_height, canvas_width = image_size
                        #     ref_height, ref_width = ref_img.shape[-2:]
                        #     # make it change to black background since we hope to keep reference image
                        #     white_canvas = -torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
                        #     scale = min(canvas_height / ref_height, canvas_width / ref_width)
                        #     new_height = int(ref_height * scale)
                        #     new_width = int(ref_width * scale)
                        #     resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                        #     top = (canvas_height - new_height) // 2
                        #     left = (canvas_width - new_width) // 2
                        #     white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                        #     ref_img = white_canvas
                        # ref_img = self._resize_for_rectangle_crop(ref_img, target_shape, reshape_mode="center")
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images, ori_image_sizes

    def load_video(self, video_path):
        decord.bridge.set_bridge("torch")

        video_reader = decord.VideoReader(uri=video_path, ctx=cpu(0))
        video_num_frames = len(video_reader)

        video = video_reader.get_batch(list(range(video_num_frames)))
        video = video.permute(3, 0, 1, 2).contiguous()
        video = video.float().div_(127.5).sub_(1.)

        return video #c f h w

    def _resize_for_rectangle_crop(
        self, arr, image_size, reshape_mode=None
    ):
        target_image_size = image_size
        # if arr.shape[3] / arr.shape[2] < target_image_size[1] / target_image_size[0]:
        arr = resize(
            arr,
            size=[
                target_image_size[0],
                int(arr.shape[3] * target_image_size[0] / arr.shape[2]),
            ],
            interpolation=InterpolationMode.BILINEAR,
        )
        if arr.shape[3]<target_image_size[1]:
            pad_left = (target_image_size[1] - arr.shape[3]) // 2
            pad_right = (
                target_image_size[1] - arr.shape[3] - pad_left
            )  # Ensure total width reaches 720
            # Apply padding (pad format: (right, left, top, bottom))
            arr = F.pad(arr, (pad_left, pad_right, 0, 0), mode="constant", value=-1)

        elif arr.shape[3]>target_image_size[1]:
            h, w = arr.shape[2], arr.shape[3]

            delta_h = h - target_image_size[0]
            delta_w = w - target_image_size[1]

            if reshape_mode == "random" or reshape_mode == "none":
                top = np.random.randint(0, delta_h + 1)
                left = np.random.randint(0, delta_w + 1)
            elif reshape_mode == "center":
                top, left = delta_h // 2, delta_w // 2
            elif reshape_mode == "top":
                top, left = 32, 0

            else:
                raise NotImplementedError
            arr = TT.functional.crop(
                arr, top=top, left=left, height=target_image_size[0], width=target_image_size[1]
            )
        return arr

    def decode_latent(self, zs, ref_images=None):
        if ref_images is None:
            ref_images = [None] * len(zs)
        else:
            assert len(zs) == len(ref_images)

        trimed_zs = []
        for z, refs in zip(zs, ref_images):
            if refs is not None:
                z = z[:, len(refs):, :, :]
            trimed_zs.append(z)

        return self.vae.decode(trimed_zs)



    def generate(self,
                 input_prompt,
                 input_frames,
                 skeleton_frames,
                 input_masks,
                 input_ref_images,
                 audio_emb=None,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 cached_context=None,
                 cached_context_null=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # Use cached context if provided, otherwise encode with T5
        if cached_context is not None and cached_context_null is not None:
            context = [t.to(self.device) for t in cached_context]
            context_null = [t.to(self.device) for t in cached_context_null]
        elif not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # vace context encode
        skeleton_frames = torch.stack(skeleton_frames, dim=0)
        num_frame = skeleton_frames.shape[2]
        skeleton_frames = rearrange(skeleton_frames, "b c f h w -> (b f) c 1 h w")
        skeleton_latents = self.vae.encode(skeleton_frames)
        skeleton_latents = torch.stack(skeleton_latents, dim=0)
        skeleton_latents = rearrange(skeleton_latents, "(b f) c 1 h w -> b c f h w",f=num_frame)

        z0, ref_latent_for_noise = self.vace_encode_frames(
            input_frames, input_ref_images, masks=input_masks
        )
        m0 = self.vace_encode_masks(input_masks, input_ref_images)
        z = self.vace_latent(z0, m0)

        target_shape = list(z0[0].shape)
        target_shape[0] = int(target_shape[0] / 2)
        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                latent_model_input[0][:, :1, :, :] = ref_latent_for_noise[0]
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input,
                    t=timestep,
                    vace_context=z,
                    vace_context_scale=context_scale,
                    skeleton_latents=skeleton_latents,
                    audio_emb=audio_emb,
                    **arg_c,
                )[0]
                noise_pred_uncond = self.model(
                    latent_model_input,
                    t=timestep,
                    vace_context=z,
                    vace_context_scale=context_scale,
                    skeleton_latents=skeleton_latents,
                    audio_emb=torch.zeros_like(audio_emb),
                    **arg_null,
                )[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.decode_latent(x0, input_ref_images)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None