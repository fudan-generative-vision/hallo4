import torch
import torch.nn as nn

from einops import rearrange


class AudioProjModel(nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
            seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_embeds):
        # merge
        video_length = audio_embeds.shape[1]
        assert video_length % 4 == 1

        audio_first, audio_rest = audio_embeds[:, :1], audio_embeds[:, 1:]
        audio_first = audio_first.repeat(1, 4, 1, 1, 1)
        audio_embeds = torch.cat([audio_first, audio_rest], dim=1)
        video_length = audio_embeds.shape[1]
        assert video_length % 4 == 0

        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        context_tokens = self.norm(context_tokens)   
        df = video_length // 4
        context_tokens = rearrange(
            context_tokens, 
            "(bz df four) m c -> bz df (four m) c", 
            df=df, 
            four=4
        )

        return context_tokens