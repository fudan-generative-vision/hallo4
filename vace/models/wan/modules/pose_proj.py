import torch
import torch.nn as nn

from einops import rearrange

import torch
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from typing import Optional

from torch import nn


class PoseProjector(ModelMixin):
    """
    A PyTorch model for projecting skeleton embeddings into a target feature space.

    This class processes multi-dimensional skeleton embedding data using a series of linear transformations,
    normalization, and activation functions. It is designed to handle temporal and spatial patching of the input data.

    Attributes:
        linear1 (torch.nn.Linear): First linear transformation layer.
        linear2 (torch.nn.Linear): Second linear transformation layer, doubling the feature dimension.
        linear3 (torch.nn.Linear): Third linear transformation layer, projecting back to the target feature dimension.
        patch_size (int): Size of the spatial patches to divide the input.
        patch_size_t (int): Size of the temporal patches to divide the input.
        norm (torch.nn.LayerNorm): Layer normalization applied to the output features.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        patch_size,
        patch_size_t: Optional[int] = None,
    ):
        """
        Initializes the PoseProjector.

        Args:
            in_dim (int): Number of input features.
            out_dim (int): Number of output features.
            patch_size (int): Size of spatial patches to divide the input.
            patch_size_t (int): Size of temporal patches to divide the input.
        """
        super().__init__()
        assert out_dim % 4 == 0, f"out_dim must be divisible by 4, but got {out_dim}"
        if patch_size_t is None:
            self.proj = nn.Conv2d(
                in_dim,
                out_dim//4,
                kernel_size=(patch_size, patch_size),
                stride=patch_size,
                bias=True,
            )
        else:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_dim, out_features=out_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(in_features=out_dim, out_features=2 * out_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(in_features=2 * out_dim, out_features=out_dim),
            )
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.norm = torch.nn.LayerNorm(out_dim// 4)

    def forward(self, skeleton_embeds: torch.Tensor):
        """
        Forward pass of the PoseProjector.

        This method processes the input skeleton embeddings through reshaping, patching,
        linear transformations, activation, and normalization.

        Args:
            skeleton_embeds (torch.Tensor):
                Input skeleton embeddings of shape (batch_size, channels, num_frames , height, width).

        Returns:
            torch.Tensor: Projected embeddings of shape (batch_size, num_patches, out_dim).
        """
        assert skeleton_embeds.shape[2] % 4 == 1
        skeleton_embeds = torch.cat([skeleton_embeds[:, :, :1].repeat(1, 1, 3, 1, 1), skeleton_embeds], dim=2)
        batch_size, channels, num_frames, height, width = skeleton_embeds.shape

        if self.patch_size_t is None:
            skeleton_embeds = skeleton_embeds.permute(0,2,1,3,4)
            skeleton_embeds = skeleton_embeds.reshape(-1, channels, height, width)
            skeleton_embeds = self.proj(skeleton_embeds)
            skeleton_embeds = skeleton_embeds.view(
                batch_size, num_frames, *skeleton_embeds.shape[1:]
            )
            skeleton_embeds = skeleton_embeds.flatten(3).transpose(
                2, 3
            )  # [batch, num_frames, height x width, channels]
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            skeleton_embeds = skeleton_embeds.permute(0, 1, 3, 4, 2)
            skeleton_embeds = skeleton_embeds.reshape(
                batch_size,
                num_frames // p_t,
                p_t,
                height // p,
                p,
                width // p,
                p,
                channels,
            )
            skeleton_embeds = (
                skeleton_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6)
                .flatten(4, 7)
                .flatten(1, 3)
            )
            skeleton_embeds = self.proj(skeleton_embeds)
        skeleton_embeds = self.norm(skeleton_embeds)
        skeleton_embeds = rearrange(skeleton_embeds, "b (df four) n d -> b df n (four d)", df = num_frames//4, four=4)
        return skeleton_embeds
