# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from torch import amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from pathlib import Path
from safetensors.torch import load_file

from .wan_model import WanModel, WanAttentionBlock, sinusoidal_embedding_1d
from .pose_proj import PoseProjector
from .audio_proj import AudioProjModel

from icecream import ic

class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0,
            enable_skeleton_cross_attn=False
    ):
        super().__init__(
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size,
            qk_norm,
            cross_attn_norm,
            eps,
            enable_skeleton_cross_attn=enable_skeleton_cross_attn,
        )
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c
    
    
class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None,
        enable_skeleton_cross_attn=False,
        enable_audio_cross_attn=False
    ):
        super().__init__(
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size,
            qk_norm,
            cross_attn_norm,
            eps,
            enable_skeleton_cross_attn,
            enable_audio_cross_attn,
        )
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x
    
    
class VaceWanModel(WanModel, nn.Module):
    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 enable_audio_cross_attn=False,
                 enable_skeleton_cross_attn=False,
                 use_gradient_checkpointing=True):
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        self.enable_skeleton_cross_attn = enable_skeleton_cross_attn
        self.enable_audio_cross_attn = enable_audio_cross_attn

        self.use_gradient_checkpointing = use_gradient_checkpointing

        # blocks
        self.blocks = nn.ModuleList(
            [
                BaseWanAttentionBlock(
                    "t2v_cross_attn",
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    block_id=self.vace_layers_mapping[i]
                    if i in self.vace_layers
                    else None,
                    enable_skeleton_cross_attn=self.enable_skeleton_cross_attn,
                    enable_audio_cross_attn=self.enable_audio_cross_attn,
                )
                for i in range(self.num_layers)
            ]
        )

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        # skeleton proj
        if self.enable_skeleton_cross_attn:
            self.pose_proj = PoseProjector(
                in_dim=in_dim, out_dim=self.dim, patch_size=self.patch_size[1]
            )
        # audio proj
        if self.enable_audio_cross_attn:
            self.audio_proj = AudioProjModel(output_dim=self.dim)

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs
    ):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        # grad checkpointing
        def create_custom_forward(module):
            def custom_forward(c, kwargs):  
                return module(c, **kwargs)
            return custom_forward

        for block in self.vace_blocks:
            if self.training and self.use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            c, new_kwargs,
                            use_reentrant=False,
                    )
            else:
                c = block(c, **new_kwargs)

        hints = torch.unbind(c)[:-1]
        return hints

    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
        skeleton_latents=None,
        audio_emb=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast("cuda",dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens
        )

        # grad checkpointing
        def create_custom_forward(module):
            def custom_forward(x, kwargs):  
                return module(x, **kwargs)
            return custom_forward

        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale

        if self.enable_skeleton_cross_attn:
            skeleton_latents = self.pose_proj(skeleton_latents)
            skeleton_latents = torch.cat([torch.zeros_like(skeleton_latents[:, :1]), skeleton_latents], dim=1)
            kwargs["skeleton_latents"] = skeleton_latents
        if self.enable_audio_cross_attn:
            audio_emb = self.audio_proj(audio_emb)
            audio_emb = torch.cat([torch.zeros_like(audio_emb[:, :1]), audio_emb], dim=1)
            kwargs['audio_emb'] = audio_emb

        for block in self.blocks:
            if self.training and self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, kwargs,
                            use_reentrant=False,
                    )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
    
    @classmethod
    def from_ckpt(
        cls,
        model_path,
        audio_model_path=None,
        additional_kwargs={},
        config_path="pretrained_models/Wan2.1_Encoders/config.json"
    ):
        config = cls.load_config(config_path)
        model = cls.from_config(config, **additional_kwargs)

        print("Loading model from", model_path)

        if model_path.endswith('.safetensors'):
            state_dict = load_file(
                model_path, device="cpu"
            )
        elif model_path.endswith('.ckpt'):
            checkpoint = torch.load(model_path, map_location="cpu",weights_only=True)
            state_dict = checkpoint.get("state_dict", checkpoint)
            del checkpoint

            if audio_model_path is not None:
                print("Loading audio model from", audio_model_path)
                audio_state_dict = torch.load(audio_model_path, map_location="cpu",weights_only=True)
                audio_state_dict = audio_state_dict.get("state_dict", audio_state_dict)
                # Loop through the old state_dict
                for key, value in audio_state_dict.items():
                    if 'audio_proj' in key:
                        # Copy as-is
                        state_dict[key] = value
                    elif 'audio_attn' in key:
                        # Rename 'audio_attn' to 'audio_cross_attn'
                        new_key = key.replace('audio_attn', 'audio_cross_attn')
                        state_dict[new_key] = value
                    elif 'audio_norm' in key:
                        new_key = key.replace('audio_norm', 'audio_cross_norm')
                        state_dict[new_key] = value
                del audio_state_dict
        else:
            raise FileNotFoundError(
                f"error weights file found in {model_path}")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    
        return model