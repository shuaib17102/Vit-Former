# ------------------------------------------------------------
# Copyright (c) University of Macau,
# Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by FuChen Zheng(YC379501)
#
# This file is the original SMAFormer implementation you uploaded,
# with two changes so it can be imported cleanly as a library module:
#   1. Removed the top-level `argparse` CLI parsing and the `ptflops`
#      import (not needed to use the model as a module, and ptflops
#      may not be installed in every environment).
#   2. Fixed a latent bug in RowAttention/ColAttention: `self.gamma`
#      referenced `self.device`, which is never defined anywhere in
#      the class -> `nn.Module` has no `.device` attribute, so
#      instantiating either class would raise AttributeError. Neither
#      class is actually used by `SMAFormer.forward`, so this never
#      surfaced -- but it's a landmine if you ever wire them in.
# No architectural logic was changed.
# ------------------------------------------------------------

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import Softmax
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
class SDPAMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, Lq, C = query.shape
        Lk = key.shape[1]

        # Add .contiguous() to prevent silent fallback to naive math backend
        q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # Restrict allowlist to memory-efficient attention
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, C)
        out = self.out_proj(attn_out)
        return out, None

class RowAttention(nn.Module):
    """Unused by SMAFormer.forward (kept for parity with the original file)."""

    def __init__(self, in_dim, q_k_dim):
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))  # fixed: was `.to(self.device)` (undefined attr)

    def forward(self, x):
        b, _, h, w = x.size()

        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        row_attn = torch.bmm(Q, K)
        row_attn = self.softmax(row_attn)
        out = torch.bmm(V, row_attn.permute(0, 2, 1))
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)
        out = self.gamma * out + x
        return out


class ColAttention(nn.Module):
    """Unused by SMAFormer.forward (kept for parity with the original file)."""

    def __init__(self, in_dim, q_k_dim):
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))  # fixed: was `.to(self.device)` (undefined attr)

    def forward(self, x):
        b, _, h, w = x.size()

        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)

        col_attn = torch.bmm(Q, K)
        col_attn = self.softmax(col_attn)
        out = torch.bmm(V, col_attn.permute(0, 2, 1))
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)
        out = self.gamma * out + x
        return out


class Modulator(nn.Module):
    def __init__(self, in_ch, out_ch, with_pos=True):
        super(Modulator, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.rate = [1, 6, 12, 18]
        self.with_pos = with_pos
        self.patch_size = 2
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA_fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // 16, in_ch, bias=False),
            nn.Sigmoid(),
        )

        # Pixel Attention
        self.PA_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.PA_bn = nn.BatchNorm2d(in_ch)
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.SA_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=rate, dilation=rate),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
            ) for rate in self.rate
        ])
        self.SA_out_conv = nn.Conv2d(len(self.rate) * out_ch, out_ch, 1)

        self.output_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self._init_weights()

        self.pj_conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.patch_size + 1,
                                  stride=self.patch_size, padding=self.patch_size // 2)
        self.pos_conv = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1, groups=self.out_ch, bias=True)
        self.layernorm = nn.LayerNorm(self.out_ch, eps=1e-6)

    def forward(self, x):
        res = x
        pa = self.PA(x)
        ca = self.CA(x)

        pa_ca = torch.softmax(pa @ ca, dim=-1)
        sa = self.SA(x)
        out = pa_ca @ sa
        out = self.norm(self.output_conv(out))
        out = out + self.bias
        synergistic_attn = out + res
        return synergistic_attn

    def PE(self, x):
        proj = self.pj_conv(x)
        pos = proj
        if self.with_pos:
            pos = proj * self.sigmoid(self.pos_conv(proj))
        pos = pos.flatten(2).transpose(1, 2)  # BCHW -> BNC
        embedded_pos = self.layernorm(pos)
        return embedded_pos

    def PA(self, x):
        attn = self.PA_conv(x)
        attn = self.PA_bn(attn)
        attn = self.sigmoid(attn)
        return x * attn

    def CA(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.CA_fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def SA(self, x):
        sa_outs = [block(x) for block in self.SA_blocks]
        sa_out = torch.cat(sa_outs, dim=1)
        sa_out = self.SA_out_conv(sa_out)
        return sa_out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SMA(nn.Module):
    def __init__(self, feature_size, num_heads, dropout):
        super(SMA, self).__init__()
        self.attention = SDPAMultiheadAttention(feature_size, num_heads, dropout)
        self.combined_modulator = Modulator(feature_size, feature_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, value, key, query):
        MSA = self.attention(query, key, value)[0]
        batch_size, seq_len, feature_size = MSA.shape
        MSA = MSA.permute(0, 2, 1).view(batch_size, feature_size, int(seq_len ** 0.5), int(seq_len ** 0.5))
        synergistic_attn = self.combined_modulator.forward(MSA)
        x = synergistic_attn.view(batch_size, feature_size, -1).permute(0, 2, 1)
        return x


class MSA(nn.Module):
    def __init__(self, feature_size, num_heads, dropout):
        super(MSA, self).__init__()
        self.attention = SDPAMultiheadAttention(feature_size, num_heads, dropout)
        self.combined_modulator = Modulator(feature_size, feature_size)

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        return attention


class E_MLP(nn.Module):
    def __init__(self, feature_size, forward_expansion, dropout):
        super(E_MLP, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, forward_expansion * feature_size),
            nn.GELU(),
            nn.Linear(forward_expansion * feature_size, feature_size),
        )
        self.linear1 = nn.Linear(feature_size, forward_expansion * feature_size)
        self.act = nn.GELU()
        self.depthwise_conv = nn.Conv2d(
            in_channels=forward_expansion * feature_size,
            out_channels=forward_expansion * feature_size,
            kernel_size=3, padding=1, groups=1,
        )
        self.pixelwise_conv = nn.Conv2d(
            in_channels=forward_expansion * feature_size,
            out_channels=forward_expansion * feature_size,
            kernel_size=3, padding=1,
        )
        self.linear2 = nn.Linear(forward_expansion * feature_size, feature_size)

    def forward(self, x):
        b, hw, c = x.size()
        feature_size = int(math.sqrt(hw))

        x = self.linear1(x)
        x = self.act(x)
        x = rearrange(x, 'b (h w) (c) -> b c h w', h=feature_size, w=feature_size)
        x = self.depthwise_conv(x)
        x = self.pixelwise_conv(x)
        x = rearrange(x, 'b c h w -> b (h w) (c)', h=feature_size, w=feature_size)
        out = self.linear2(x)
        return out


class SMAFormerBlock(nn.Module):
    def __init__(self, ch_in, ch_out, heads, dropout, forward_expansion, fusion_gate):
        super(SMAFormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(ch_out)
        self.norm2 = nn.LayerNorm(ch_out)
        self.MSA = MSA(ch_out, heads, dropout)
        self.synergistic_multi_attention = SMA(ch_out, heads, dropout)
        self.e_mlp = E_MLP(ch_out, forward_expansion, dropout)
        self.fusion_gate = fusion_gate
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, value, key, query, res):
        if self.fusion_gate:
            attention = self.synergistic_multi_attention(query, key, value)
        else:
            attention = self.MSA(query, key, value)
        query = self.dropout(self.norm1(attention + res))
        feed_forward = self.e_mlp(query)
        out = self.dropout(self.norm2(feed_forward + query))
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, heads, dropout, forward_expansion, num_layers, fusion_gate):
        super(EncoderBlock, self).__init__()
        self.layers = nn.ModuleList([
            SMAFormerBlock(in_ch, out_ch, heads, dropout, forward_expansion, fusion_gate) for _ in range(num_layers)
        ])
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, res):
        for layer in self.layers:
            x = layer(res, res, x, x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, heads, dropout, forward_expansion, num_layers, fusion_gate):
        super(DecoderBlock, self).__init__()
        self.layers = nn.ModuleList([
            SMAFormerBlock(in_ch, out_ch, heads, dropout, forward_expansion, fusion_gate) for _ in range(num_layers)
        ])
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, res):
        for layer in self.layers:
            x = layer(res, res, x, x)
        return x


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()
        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample_Transpose(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample_Transpose, self).__init__()
        self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.upsample(x)


class Cross_AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(Cross_AttentionBlock, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
        )
        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )
        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class SMAFormer(nn.Module):
    """Original author architecture, unchanged, except `n_classes` and
    `in_channels` are now constructor args instead of hardcoded to 3."""

    def __init__(self, args=None, in_channels=3, n_classes=3):
        super(SMAFormer, self).__init__()
        self.args = args
        patch_size = 2
        filters = [16, 32, 64, 128, 256, 512]
        encoder_layer = 1
        decoder_layer = 1
        self.patch_size = patch_size
        self.filters = filters
        self.n_classes = n_classes

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.patch_embedding1 = Modulator(in_ch=filters[0], out_ch=filters[1])
        self.EncoderBlock1 = EncoderBlock(in_ch=filters[1], out_ch=filters[1], heads=8, dropout=0.1,
                                           forward_expansion=2, num_layers=encoder_layer, fusion_gate=True)
        self.residual_conv1 = ResidualConv(filters[1], filters[2], 2, 1)

        self.patch_embedding2 = Modulator(in_ch=filters[2], out_ch=filters[3])
        self.EncoderBlock2 = EncoderBlock(in_ch=filters[3], out_ch=filters[3], heads=8, dropout=0.1,
                                           forward_expansion=2, num_layers=encoder_layer, fusion_gate=True)
        self.residual_conv2 = ResidualConv(filters[3], filters[4], 2, 1)

        self.patch_embedding3 = Modulator(in_ch=filters[4], out_ch=filters[5])
        self.EncoderBlock3 = EncoderBlock(in_ch=filters[5], out_ch=filters[5], heads=8, dropout=0.1,
                                           forward_expansion=2, num_layers=encoder_layer, fusion_gate=True)
        self.EncoderBlock4 = EncoderBlock(in_ch=filters[5], out_ch=filters[5], heads=8, dropout=0.1,
                                           forward_expansion=2, num_layers=encoder_layer, fusion_gate=True)

        self.DecoderBlock1 = DecoderBlock(in_ch=filters[5], out_ch=filters[5], heads=8, dropout=0.1,
                                           forward_expansion=2, num_layers=decoder_layer, fusion_gate=True)

        self.upsample = Upsample_(2)
        self.upsample_transpose1 = Upsample_Transpose(filters[5], filters[4], kernel=2, stride=2)
        self.DecoderBlock2 = DecoderBlock(in_ch=filters[5], out_ch=filters[5], heads=8, dropout=0.,
                                           forward_expansion=2, num_layers=decoder_layer, fusion_gate=True)

        self.upsample_transpose2 = Upsample_Transpose(filters[5], filters[4], kernel=2, stride=2)
        self.upsample_transpose3 = Upsample_Transpose(filters[4] + filters[3], filters[3], kernel=1, stride=1)
        self.DecoderBlock3 = DecoderBlock(in_ch=filters[3], out_ch=filters[3], heads=8, dropout=0.1,
                                           forward_expansion=2, num_layers=decoder_layer, fusion_gate=True)

        self.upsample_transpose4 = Upsample_Transpose(filters[3], filters[2], kernel=2, stride=2)
        self.upsample_transpose5 = Upsample_Transpose(filters[3], filters[2], kernel=2, stride=2)
        self.DecoderBlock4 = DecoderBlock(in_ch=filters[2], out_ch=filters[2], heads=8, dropout=0.1,
                                           forward_expansion=2, num_layers=decoder_layer, fusion_gate=True)
        self.adjust = Upsample_Transpose(filters[1], filters[2], kernel=1, stride=1)
        self.upsample_transpose6 = Upsample_Transpose(filters[2], filters[1], kernel=2, stride=2)
        self.output_layer1 = nn.Sequential(nn.Conv2d(filters[1], filters[0], 1))
        self.output_layer2 = nn.Sequential(nn.Conv2d(filters[0], n_classes, 1))

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.patch_embedding1.PE(x1)
        e1 = self.EncoderBlock1(x2, x2)
        b, num_patch, c = e1.size()
        x2 = e1.view(b, c, int(num_patch ** 0.5), int(num_patch ** 0.5))
        x2 = self.residual_conv1(x2)

        x3 = self.patch_embedding2.PE(x2)
        e2 = self.EncoderBlock2(x3, x3)
        b, num_patch, c = e2.size()
        e2 = e2.view(b, c, num_patch // self.filters[2], num_patch // self.filters[2])
        x3 = self.residual_conv2(e2)

        x4 = self.patch_embedding3.PE(x3)
        e3 = self.EncoderBlock3(x4, x4)
        e4 = self.EncoderBlock4(e3, e3)

        x5 = self.DecoderBlock1(e4, e4)
        b, hw, c = x5.size()
        h = w = int(hw ** 0.5)
        x5 = x5.contiguous().permute(0, 2, 1).view(b, c, h, w)
        x6 = self.upsample_transpose1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        b, c, h, w = x6.size()
        x6 = x6.view(b, c, h * w).contiguous().permute(0, 2, 1)
        b, num_patch, c = e3.size()
        e3 = e3.view(b, c, int(num_patch ** 0.5), int(num_patch ** 0.5))
        e3 = self.upsample(e3)
        b, c, h, w = e3.size()
        e3 = e3.view(b, c, h * w).contiguous().permute(0, 2, 1)
        x6 = self.DecoderBlock2(x6, e3)
        b, hw, c = x6.size()
        h = w = int(hw ** 0.5)
        x6 = x6.permute(0, 2, 1).contiguous().view(b, c, h, w)

        x7 = self.upsample_transpose2(x6)
        x7 = torch.cat([x7, e2], dim=1)
        x7 = self.upsample_transpose3(x7)
        b, c, h, w = x7.size()
        x7 = x7.view(b, c, h * w).contiguous().permute(0, 2, 1)
        b, c, h, w = e2.size()
        e2 = e2.view(b, c, h * w).contiguous().permute(0, 2, 1)
        x7 = self.DecoderBlock3(x7, e2)
        b, hw, c = x7.size()
        h = w = int(hw ** 0.5)
        x7 = x7.permute(0, 2, 1).contiguous().view(b, c, h, w)

        x8 = self.upsample_transpose4(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.upsample_transpose5(x8)
        b, c, h, w = x8.size()
        x8 = x8.view(b, c, h * w).contiguous().permute(0, 2, 1)
        b_e1, hw_e1, c_e1 = e1.size()
        h_e1 = w_e1 = int(hw_e1 ** 0.5)
        e1 = e1.permute(0, 2, 1).contiguous().view(b_e1, c_e1, h_e1, w_e1)
        e1 = self.adjust(e1)
        b_e1, c_e1, h_e1, w_e1 = e1.size()
        e1 = e1.view(b_e1, c_e1, h_e1 * w_e1).contiguous().permute(0, 2, 1)
        x8 = self.DecoderBlock4(x8, e1)
        b, hw, c = x8.size()
        h = w = int(hw ** 0.5)
        x8 = x8.permute(0, 2, 1).contiguous().view(b, c, h, w)
        x8 = self.upsample_transpose6(x8)

        out = self.output_layer1(x8)
        out = self.output_layer2(out)
        return out
