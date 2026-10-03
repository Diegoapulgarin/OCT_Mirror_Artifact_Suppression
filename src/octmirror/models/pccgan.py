"""PC-CGAN architectures: pix2pix U-Net generator and PatchGAN discriminator.

Both networks operate on 2-channel (real, imag) normalized complex B-scans.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):

    def __init__(self, in_ch: int = 2, out_ch: int = 2, use_dropout: bool = True):
        super().__init__()
        self.use_dropout = use_dropout

        def enc_block(in_c, out_c, apply_bn=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not apply_bn)]
            if apply_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def dec_block(in_c, out_c, apply_dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            if apply_dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder (e1 without BN)
        self.e1 = enc_block(in_ch, 64, apply_bn=False)
        self.e2 = enc_block(64, 128)
        self.e3 = enc_block(128, 256)
        self.e4 = enc_block(256, 512)
        self.e5 = enc_block(512, 512)
        self.e6 = enc_block(512, 512)
        # Bottleneck (stride-2 conv + ReLU, without BN)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # Decoder (concatenated skips)
        self.d1 = dec_block(512, 512, apply_dropout=use_dropout)      # cat with e6
        self.d2 = dec_block(512+512, 512, apply_dropout=use_dropout)  # cat with e5
        self.d3 = dec_block(512+512, 512, apply_dropout=use_dropout)  # cat with e4
        self.d4 = dec_block(512+512, 256, apply_dropout=False)        # cat with e3
        self.d5 = dec_block(256+256, 128, apply_dropout=False)        # cat with e2
        self.d6 = dec_block(128+128, 64, apply_dropout=False)         # cat with e1

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(64+64, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()   # Output in [0,1] to match normalized log-scale representation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        b = self.bottleneck(e6)

        d1 = self.d1(b)
        d1 = torch.cat([d1, e6], dim=1)
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e5], dim=1)
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e4], dim=1)
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e3], dim=1)
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e2], dim=1)
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e1], dim=1)
        out = self.out_conv(d6)
        return out


class PatchGANDiscriminator(nn.Module):

    def __init__(self, in_ch: int = 2):
        super().__init__()
        ch_in = in_ch * 2  # concat real + fake

        def disc_block(in_c, out_c, stride, apply_bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=not apply_bn)]
            if apply_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.c1 = disc_block(ch_in, 64, stride=2, apply_bn=False)
        self.c2 = disc_block(64, 128, stride=2)
        self.c3 = disc_block(128, 256, stride=2)
        self.c4 = disc_block(256, 512, stride=2)
        self.c5 = disc_block(512, 512, stride=1)
        self.out = nn.Conv2d(512, 1, 4, stride=1, padding=1)

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Score a pair concatenated along channels.

        During training ``real`` is the conditioning (mirror-corrupted) input and
        ``fake`` is either the clean target or the generator output.
        """
        x = torch.cat([real, fake], dim=1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        return self.out(x)
