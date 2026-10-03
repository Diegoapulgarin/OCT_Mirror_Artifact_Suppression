"""Conditional DDPM: U-Net with time embeddings and self-attention, and the noise schedule.

The U-Net receives the noisy sample and the condition (mirror-corrupted ROI), each with
2 channels (real, imag), concatenated into 4 input channels, and predicts the 2-channel
noise.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel

# Same backend set (and order) that torch.backends.cuda.sdp_kernel(enable_flash=True,
# enable_math=True, enable_mem_efficient=True) enables; see tests/test_models.py.
_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
    SDPBackend.CUDNN_ATTENTION,
]


class SinusoidalEmbeddings(nn.Module):
    """Fixed sinusoidal time-step embeddings, returned as (B, embed_dim, 1, 1)."""

    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.register_buffer('embeddings', embeddings)  # buffer, not a parameter

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings.to(x.device)[t]
        return embeds[:, :, None, None]


class ResBlock(nn.Module):
    """Residual block (GroupNorm-ReLU-Conv x2) with additive time embedding."""

    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x


class Attention(nn.Module):
    """Multi-head self-attention over spatial positions using fused SDPA kernels."""

    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        # contiguous() required: fused kernels need stride-1 last dim, rearrange+indexing breaks it
        q, k, v = x[0].contiguous(), x[1].contiguous(), x[2].contiguous()

        # Flash / memory-efficient attention reduce memory from O(N^2) to O(N);
        # the math backend is kept as a fallback when they do not apply.
        with sdpa_kernel(_SDPA_BACKENDS):
            x = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=False,
                dropout_p=self.dropout_prob if self.training else 0.0
            )
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')


class UnetLayer(nn.Module):
    """Two residual blocks, optional attention, and a down- or up-sampling convolution."""

    def __init__(self,
                 upscale: bool,
                 attention: bool,
                 num_groups: int,
                 dropout_prob: float,
                 num_heads: int,
                 C: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x


class UNET(nn.Module):
    """Conditional diffusion U-Net.

    ``forward(x, t, condition)`` concatenates the noisy sample ``x`` (B, 2, H, W) and
    the ``condition`` (B, 2, H, W) into ``input_channels=4`` channels and returns the
    predicted noise (B, ``output_channels``, H, W).
    """

    def __init__(self,
                 Channels: List = [64, 128, 256, 512, 512, 384],
                 Attentions: List = [False, False, False, True, True, True],
                 Upscales: List = [False, False, False, True, True, True],
                 num_groups: int = 32,
                 dropout_prob: float = 0.1,
                 num_heads: int = 2,
                 input_channels: int = 4,
                 output_channels: int = 2,
                 time_steps: int = 1000):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1]//2)+Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, condition), dim=1)
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))


class DDPMScheduler(nn.Module):
    """Linear beta schedule (1e-4 to 0.02) and cumulative alpha products.

    Args:
        num_time_steps: Number of diffusion steps.
        device: Device holding ``beta`` and ``alpha``.
    """

    def __init__(self, num_time_steps: int = 1000, device="cpu"):
        super().__init__()
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False).to(self.device)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).to(self.device)

    def forward(self, t):
        return self.beta[t], self.alpha[t]
