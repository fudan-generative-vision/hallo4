# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import logging
import math
import os
import random
import sys
import time

import torch
import torch.distributed as dist

from vace.models.utils.preprocessor import AudioProcessor
from vace.models.utils.utils import add_audio, cache_image, cache_video, str2bool
from vace.models.wan import WanVace
from vace.models.wan.configs import (
    SIZE_CONFIGS,
    SUPPORTED_SIZES,
    WAN_CONFIGS,
)


EXAMPLE_PROMPT = {
    "vace-1.3B": {
        "src_ref_images": "/cpfs01/projects-HDD/cfff-721febfbdfb0_HDD/public/champ/workspaces/xumingw/wyd-benchmark/processed_select_480x832_frames/-irXG0K3Wfo.mp4/0001.png",
        "src_video": "/cpfs01/projects-HDD/cfff-721febfbdfb0_HDD/public/champ/workspaces/xumingw/wyd-benchmark/processed_dwpose_select_480x832/-irXG0K3Wfo.mp4",
        "prompt": "A man wearing a white shirt is tying a tie around his neck.",
    }
}

def random_mask_patches(x, patch_size=2, mask_ratio=0.1):
    """
    Randomly masks 2x2 patches in a (C, H, W) tensor by setting them to the min value.

    Args:
        x (torch.Tensor): Input tensor of shape (C, H, W).
        patch_size (int): Size of square patches.
        mask_ratio (float): Fraction of patches to mask.

    Returns:
        torch.Tensor: Tensor with masked patches set to the minimum value of x.
    """
    C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w

    # Generate patch-level mask
    num_mask = int(mask_ratio * total_patches)
    patch_mask = torch.zeros(total_patches, dtype=torch.bool, device=x.device)
    patch_mask[torch.randperm(total_patches, device=x.device)[:num_mask]] = True
    patch_mask = patch_mask.view(num_patches_h, num_patches_w)

    # Expand patch mask to pixel level
    mask = patch_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    mask = mask.unsqueeze(0).expand(C, -1, -1)  # (C, H, W)

    # Apply min value masking
    x_masked = x.clone()
    x_masked[mask] = -1.0

    return x_masked

def read_from_file(p, rank=0, world_size=1,save_dir="outputs"):
    prompts = []
    src_videos = []
    src_ref_images = []
    audio_embs = []

    with open(p, "r") as fin:
        cnt = -1
        for line in fin:
            p, r, v, a = line.strip().split("@@")
            case_name = os.path.basename(v)[:-4]
            if os.path.exists(os.path.join(save_dir, f"{case_name}_out_video.mp4")):
                continue
        
            cnt += 1
            if cnt % world_size != rank:
                continue
            prompts.append(p)
            src_ref_images.append(r)
            src_videos.append(v)
            audio_embs.append(a)
            

    return prompts, src_ref_images, src_videos, audio_embs

def center_crop_width(tensor, original_size, target_height):
    """
    Scales width according to target_height and crops the width at center.

    Args:
        tensor (torch.Tensor): Input tensor of shape (f, c, h, w)
        original_size (tuple): (original_height, original_width)
        target_height (int): Desired height to scale to

    Returns:
        torch.Tensor: Width-center-cropped tensor
    """
    if tensor.shape[2]/ tensor.shape[3] > original_size[0] / original_size[1]:
        return tensor

    orig_h, orig_w = original_size
    scaled_w = int(orig_w * target_height / orig_h)
    start = (tensor.shape[-1] - scaled_w) // 2
    end = start + scaled_w
    return tensor[..., start:end]



def validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.model_name in WAN_CONFIGS, f"Unsupport model name: {args.model_name}"
    assert args.model_name in EXAMPLE_PROMPT, f"Unsupport model name: {args.model_name}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 25

    if args.sample_shift is None:
        args.sample_shift = 8.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 81

    args.base_seed = (
        args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    )
    # Size check
    assert args.size in SUPPORTED_SIZES[args.model_name], (
        f"Unsupport size {args.size} for model name {args.model_name}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.model_name])}"
    )
    return args


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vace-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="The model name to run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="480*832",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to sample from a image or video. The number should be 4n+1",
    )
    parser.add_argument(
        "--start_inf_frame",
        type=int,
        default=0,
        help="which frame to start inference",
    )
    parser.add_argument(
        "--n_motion_frame",
        type=int,
        default=1,
        help="How many frames use as motion frames",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="pretrained_models/Wan2.1_Encoders",
        help="The path to the checkpoint directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="The path to the trained checkpoint.",
    )
    parser.add_argument(
        "--offload_model",
        action="store_true",
        default=False,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.",
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.",
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.",
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The file to save the generated image or video to.",
    )
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.",
    )
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.",
    )
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None.",
    )
    parser.add_argument(
        "--src_audio",
        type=str,
        default=None,
        help="The file list of the source audio.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.",
    )
    parser.add_argument(
        "--use_prompt_extend",
        default="plain",
        choices=["plain", "wan_zh", "wan_en", "wan_zh_ds", "wan_en_ds"],
        help="Whether to use prompt extend.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=2025,
        help="The seed to use for generating the image or video.",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="The solver used to sample.",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps."
    )
    parser.add_argument(
        "--max_round", type=int, default=None, help="The iterately sampling round."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=6.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--audio_separator_model_path",
        type=str,
        default="./pretrained_models/audio_separator/Kim_Vocal_2.onnx",
        help="The path of audio seperator model."
    )
    parser.add_argument(
        "--wav2vec_model_path",
        type=str,
        default="./pretrained_models/wav2vec2-base-960h",
        help="The path of wav2vec model."
    )
    parser.add_argument(
        "--wav2vec_features",
        type=str,
        default="all",
        help="The features type of wav2vec model."
    )
    parser.add_argument(
        "--rank_idx",
        type=int,
        default=0,
        help="Batch index to process, you may pass machine rank as it like $RANK",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="num of batches to create like $WORLD_SIZE",
        required=False,
    )
    return parser


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)

def process_audio_emb(audio_emb):
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb

def main(args):
    args = argparse.Namespace(**args) if isinstance(args, dict) else args
    args = validate_args(args)
    rank = 0
    device = 0

    if args.use_prompt_extend and args.use_prompt_extend != "plain":
        # do not expand prompt
        prompt_expander = None

    cfg = WAN_CONFIGS[args.model_name]


    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if rank == 0:
        if args.save_dir is None:
            save_dir = os.path.join(
                "results",
                "vace_wan_1.3b",
                time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())),
            )
        else:
            save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.model_name]["prompt"]
        args.src_video = EXAMPLE_PROMPT[args.model_name].get("src_video", None)
        args.src_mask = EXAMPLE_PROMPT[args.model_name].get("src_mask", None)
        args.src_ref_images = EXAMPLE_PROMPT[args.model_name].get(
            "src_ref_images", None
        )
    logging.info(f"Input prompt: {args.prompt}")
    if args.prompt.endswith("txt"):
        prompts, src_ref_images, src_videos, audios = read_from_file(
            args.prompt,
            rank=args.rank_idx,
            world_size=args.world_size,
            save_dir=save_dir,
        )
        src_masks = [None] * len(prompts)
    else:
        prompts = [args.prompt]
        src_videos = [args.src_video]
        src_ref_images = [args.src_ref_images]
        audios = [args.src_audio]
        src_masks = [None] * len(prompts)


    logging.info("Creating WanT2V pipeline.")
    wan_vace = WanVace(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        model_path=args.model_path,
        enable_skeleton_cross_attn=True,
        enable_audio_cross_attn=True
    )

    logging.info(f"Starting Inference Cases. Total count: {len(prompts)}")

    for prompt, skeleton_video, src_mask, src_ref_image, audio_path in zip(
        prompts, src_videos, src_masks, src_ref_images, audios
    ):
        case_name = os.path.basename(skeleton_video)[:-4]
        if os.path.exists(os.path.join(save_dir, f"{case_name}_out_video.mp4")):
            continue
        logging.info(f"Processing {case_name}.")
        logging.info("Preparing Condition.")
        whole_skeleton_video, whole_src_mask, whole_src_ref_image,ref_image_sizes = wan_vace.prepare_source(
            [skeleton_video],
            [src_mask],
            [None if src_ref_image is None else src_ref_image.split(",")],
            args.frame_num,
            SIZE_CONFIGS[args.size],
            device,
        )

        whole_skeleton_video = [whole_skeleton_video[0][:,args.start_inf_frame:]]
        whole_src_mask = [whole_src_mask[0][:,args.start_inf_frame:]]
        
        if audio_path is not None:
            audio_separator_model_file = args.audio_separator_model_path
            wav2vec_model_path = args.wav2vec_model_path
            wav2vec_only_last_features = args.wav2vec_features == "last"
            audio_processor = AudioProcessor(
                16000,
                wav2vec_model_path,
                wav2vec_only_last_features,
                os.path.dirname(audio_separator_model_file),
                os.path.basename(audio_separator_model_file),
                os.path.join(".cache", "audio_preprocess")
            )
            audio_emb, length = audio_processor.preprocess(audio_path)
            whole_audio_emb = process_audio_emb(audio_emb).to(device=wan_vace.model.device, dtype=wan_vace.model.dtype)
            whole_audio_emb = whole_audio_emb[args.start_inf_frame:]

        video = []
        motion_frames = whole_src_ref_image[0][0].repeat(1, args.n_motion_frame, 1, 1)
        times = math.ceil((min(whole_skeleton_video[0].shape[1],whole_audio_emb.shape[0])-1) / (args.frame_num-1))
        if args.max_round:
            times = min(times, args.max_round)
        for i in range(times):
            logging.info(f"{i+1}/{times}")
            pad_len=0
            pad_audio_len=0

            if i == 0:
                skeleton_video = [whole_skeleton_video[0][
                    :, i * args.frame_num : (i + 1) * args.frame_num
                ]]
                clip_src_mask = [
                    whole_src_mask[0][:, i * args.frame_num : (i + 1) * args.frame_num]
                ]
                audio_tensor = whole_audio_emb[
                i * args.frame_num : (i + 1) * args.frame_num
                ]
            else:
                skeleton_video = [whole_skeleton_video[0][
                    :, i * (args.frame_num-1) : (i + 1) * (args.frame_num-1)+1
                ]]
                clip_src_mask = [
                    whole_src_mask[0][:, i * (args.frame_num-1) : (i + 1) * (args.frame_num-1)+1]
                ]
                audio_tensor = whole_audio_emb[
                i * (args.frame_num-1) : (i + 1) * (args.frame_num-1)+1
                ]

            skeleton_video[0] = wan_vace._resize_for_rectangle_crop(skeleton_video[0],SIZE_CONFIGS[args.size],reshape_mode="center")

            if skeleton_video[0].shape[1] < args.frame_num:
                pad_len = args.frame_num - skeleton_video[0].shape[1]

                last_video_frame = skeleton_video[0][:, -1:].repeat(1, pad_len,1,1)
                last_mask_frame = clip_src_mask[0][:, -1:].repeat(1, pad_len,1,1)

                skeleton_video = [torch.cat([skeleton_video[0], last_video_frame], dim=1)]
                clip_src_mask = [torch.cat([clip_src_mask[0], last_mask_frame], dim=1)]   
            
            if audio_tensor.shape[0]<args.frame_num:
                pad_audio_len = args.frame_num - audio_tensor.shape[0]
                last_audio_tensor = audio_tensor[-1:].repeat(pad_audio_len,1,1,1)
                audio_tensor = torch.cat([audio_tensor,last_audio_tensor],dim=0)


            clip_src_video = [torch.zeros_like(clip_src_mask[0])]

            # keep firsr frame
            if i>0:
                clip_src_mask[0][:, :args.n_motion_frame] = 0
                clip_src_video[0][:, :args.n_motion_frame, :, :] = random_mask_patches(motion_frames.squeeze(1)).unsqueeze(1)

            audio_tensor = audio_tensor.unsqueeze(0)
    
            logging.info("Generating video...")
            clip_video = wan_vace.generate(
                    prompt,
                    clip_src_video,
                    skeleton_video,
                    clip_src_mask,
                    whole_src_ref_image,
                    audio_emb=audio_tensor,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model,
                ).cpu()
            torch.cuda.empty_cache()
            if i==0:
                video.append(clip_video)
            else:
                video.append(clip_video[:, 1:,])
            if pad_len>0 or pad_audio_len>0:
                video[-1]=video[-1][:, :-max(pad_len,pad_audio_len)]
            motion_frames = video[-1][:, -args.n_motion_frame:, :, :]
        video = torch.concat(video, dim=1)
        video = center_crop_width(video, ref_image_sizes[0], SIZE_CONFIGS[args.size][0])

        # save debug visulinfo first
        ret_data = {}
        if rank == 0:
            save_file = os.path.join(save_dir, f"{case_name}_src_video.mp4")
            cache_video(
                tensor=whole_skeleton_video[0][None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            logging.info(f"Saving src_video to {save_file}")
            ret_data["src_video"] = save_file

            save_file = os.path.join(save_dir, f"{case_name}_src_mask.mp4")
            cache_video(
                tensor=whole_src_mask[0][None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(0, 1),
            )
            logging.info(f"Saving src_mask to {save_file}")
            ret_data["src_mask"] = save_file

            if whole_src_ref_image[0] is not None:
                for i, ref_img in enumerate(whole_src_ref_image[0]):
                    save_file = os.path.join(
                        save_dir, f"{case_name}_src_ref_image_{i}.png"
                    )
                    cache_image(
                        tensor=ref_img[:, 0, ...],
                        save_file=save_file,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    logging.info(f"Saving src_ref_image_{i} to {save_file}")
                    ret_data[f"src_ref_image_{i}"] = save_file
            save_file = os.path.join(save_dir, f"{case_name}_out_video.mp4")
            cache_video(
                tensor=video[None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            logging.info(f"Saving generated video to {save_file}")
            ret_data["out_video"] = save_file

            if audio_path is not None:
                add_audio(save_file, audio_path)
        
        del whole_skeleton_video, video
        torch.cuda.empty_cache()

    logging.info("Finished.")
    return ret_data


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
