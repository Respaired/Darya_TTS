# borrowed from https://github.com/yl4579/DMOSpeech2/blob/main/src/discriminator_conformer.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
from pathlib import Path
from torchaudio.models import Conformer


class ResBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2):
        super().__init__()
        self._n_groups = 8
        self.blocks = nn.ModuleList([
            self._get_conv(hidden_dim, dilation=3**i, dropout_p=dropout_p)
            for i in range(n_conv)])


    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, dropout_p=0.2):
        layers = [
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.GroupNorm(num_groups=self._n_groups, num_channels=hidden_dim),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)

def _pool_out_length(lengths: torch.Tensor, kernel_size: int, stride: int | None = None, padding: int = 0, dilation: int = 1) -> torch.Tensor:
    if stride is None:
        stride = kernel_size
    out = torch.div(
        lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride,
        rounding_mode="floor",
    ) + 1
    return out.clamp_min(1)


class ConformerDiscirminator(nn.Module):
    def __init__(self, input_dim, channels=512, num_layers=3, num_heads=8, depthwise_conv_kernel_size=15, use_group_norm=True):
        super().__init__()

        self.input_layer = nn.Conv1d(input_dim, channels, kernel_size=3, padding=1)

        self.resblock1 = nn.Sequential(
            ResBlock(channels),
            nn.GroupNorm(num_groups=1, num_channels=channels)
        )

        self.resblock2 = nn.Sequential(
            ResBlock(channels),
            nn.GroupNorm(num_groups=1, num_channels=channels)
        )

        self.conformer1 = Conformer(
            input_dim=channels,
            num_heads=num_heads,
            ffn_dim=channels * 2,
            num_layers=1,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size // 2,
            use_group_norm=use_group_norm,
        )

        self.conformer2 = Conformer(
            input_dim=channels,
            num_heads=num_heads,
            ffn_dim=channels * 2,
            num_layers=num_layers - 1,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            use_group_norm=use_group_norm,
        )

        self.linear = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x, lengths: torch.Tensor):
        # x: list of [B, T, C]
        # lengths: [B]
        x = torch.cat(x, dim=-1)      # [B, T, C_total]
        x = x.transpose(1, 2)         # [B, C_total, T]

        x = self.input_layer(x)

        x = self.resblock1(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        lengths = _pool_out_length(lengths, kernel_size=2, stride=2)

        x = self.resblock2(x)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        lengths = _pool_out_length(lengths, kernel_size=2, stride=2)

        x = x.transpose(1, 2)         # [B, T', C]
        x, _ = self.conformer1(x, lengths)
        x, _ = self.conformer2(x, lengths)
        x = x.transpose(1, 2)         # [B, C, T']

        out = self.linear(x).squeeze(1)  # [B, T']
        return out