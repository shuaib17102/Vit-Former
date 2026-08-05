"""
SMAFormerBaseline
==================
A subclass of the original SMAFormer (smaformer_core.py) that fuses a
pretrained ViT-B/16 into the bottleneck, per the paper's stated protocol
of initializing the transformer backbone from ImageNet ViT weights.
See vit_bottleneck_branch.py for why this is a *fusion*, not a literal
state_dict load into SMAFormer's own attention blocks -- there's no
lossless mapping between a standard ViT and SMAFormer's custom
SMA/Modulator blocks, so a "clean load into the existing blocks" isn't
actually achievable without gutting the architecture. This gives you
a genuinely ViT-pretrained bottleneck while keeping every other part
of the published architecture (and its skip connections) untouched.

Only `forward` is overridden, and only to splice in the fusion at the
bottleneck; everything else is inherited from SMAFormer as-is.
"""
import math
import torch
import torch.nn as nn

from smaformer_core import SMAFormer
from vit_bottleneck_branch import ViTBottleneckBranch


class SMAFormerBaseline(SMAFormer):
    def __init__(
        self,
        args=None,
        in_channels: int = 3,
        n_classes: int = 1,           # 1 for ISIC 2018 Task 1 binary lesion mask
        img_size: int = 224,
        vit_name: str = "vit_base_patch16_224",
        pretrained_vit: bool = True,
        freeze_vit_blocks: int = 8,   # freeze first 8 of 12 blocks by default; see note below
        per_channel_gate: bool = False,  # ablation option, see vit_gate below -- default preserves current scalar behavior
    ):
        super().__init__(args=args, in_channels=in_channels, n_classes=n_classes)

        self.img_size = img_size
        self.vit_branch = ViTBottleneckBranch(
            vit_name=vit_name,
            img_size=img_size,
            target_grid=img_size // 32,            # matches SMAFormer's bottleneck grid at this resolution
            target_ch=self.filters[5], # 512
            pretrained=pretrained_vit,
            freeze_blocks=freeze_vit_blocks,
        )
        # Zero-initialized gate: at step 0 the model is mathematically
        # identical to the vanilla SMAFormer forward pass. Training
        # decides how much (if any) of the ViT prior to pull in.
        #
        # Default is a single global scalar -- the model can only learn
        # one mixing weight for the *entire* ViT contribution, it can't
        # trust some of the 512 ViT-projected channels more than others.
        # per_channel_gate=True switches to one learnable scalar per
        # channel (still zero-init, same mathematical-identity-at-step-0
        # guarantee) -- costs 512 extra parameters, worth trying as an
        # ablation on top of the scalar baseline, not a required change.
        gate_shape = (self.filters[5],) if per_channel_gate else (1,)
        self.vit_gate = nn.Parameter(torch.zeros(gate_shape))

    def load_pretrained_vit_weights(self, vit_name: str = None, freeze_vit_blocks: int = None):
        """Re-download / re-load ImageNet ViT weights into the branch,
        optionally with a different checkpoint or a different freeze
        schedule, without rebuilding the rest of SMAFormerBaseline.
        Useful for resuming an experiment or trying a different ViT
        variant (vit_base_patch16_224, vit_large_patch16_224, a DINO or
        SAM-pretrained ViT-B via a different timm model name, etc.)."""
        name = vit_name if vit_name is not None else "vit_base_patch16_224"
        freeze = freeze_vit_blocks if freeze_vit_blocks is not None else 8

        self.vit_branch = ViTBottleneckBranch(
            vit_name=name,
            img_size=self.img_size,
            target_grid=self.img_size // 32,
            target_ch=self.filters[5],
            pretrained=True,
            freeze_blocks=freeze,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.patch_embedding1.PE(x1)
        e1 = self.EncoderBlock1(x2, x2)
        b, num_patch, c = e1.size()
        x2 = e1.view(b, c, math.isqrt(num_patch), math.isqrt(num_patch))
        x2 = self.residual_conv1(x2)

        x3 = self.patch_embedding2.PE(x2)
        e2 = self.EncoderBlock2(x3, x3)
        b, num_patch, c = e2.size()
        h_grid = w_grid = math.isqrt(num_patch)
        e2 = e2.view(b, c, h_grid, w_grid)
        x3 = self.residual_conv2(e2)

        x4 = self.patch_embedding3.PE(x3)  # [B, 49, 512]  (7x7 tokens at 224 input)

        # --- ViT fusion point ---
        vit_feats = self.vit_branch(x)                              # [B, 512, 7, 7] at img_size=224
        b_v, c_v, h_v, w_v = vit_feats.shape
        vit_tokens = vit_feats.view(b_v, c_v, h_v * w_v).permute(0, 2, 1).contiguous()  # [B, 49, 512]
        x4 = x4 + self.vit_gate * vit_tokens
        # --- end fusion ---

        e3 = self.EncoderBlock3(x4, x4)
        e4 = self.EncoderBlock4(e3, e3)

        x5 = self.DecoderBlock1(e4, e4)
        b, hw, c = x5.size()
        h = w = math.isqrt(hw)
        x5 = x5.contiguous().permute(0, 2, 1).view(b, c, h, w)
        x6 = self.upsample_transpose1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        b, c, h, w = x6.size()
        x6 = x6.view(b, c, h * w).contiguous().permute(0, 2, 1)
        b, num_patch, c = e3.size()
        e3 = e3.view(b, c, math.isqrt(num_patch), math.isqrt(num_patch))
        e3 = self.upsample(e3)
        b, c, h, w = e3.size()
        e3 = e3.view(b, c, h * w).contiguous().permute(0, 2, 1)
        x6 = self.DecoderBlock2(x6, e3)
        b, hw, c = x6.size()
        h = w = math.isqrt(hw)
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
        h = w = math.isqrt(hw)
        x7 = x7.permute(0, 2, 1).contiguous().view(b, c, h, w)

        x8 = self.upsample_transpose4(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.upsample_transpose5(x8)
        b, c, h, w = x8.size()
        x8 = x8.view(b, c, h * w).contiguous().permute(0, 2, 1)
        b_e1, hw_e1, c_e1 = e1.size()
        h_e1 = w_e1 = math.isqrt(hw_e1)
        e1 = e1.permute(0, 2, 1).contiguous().view(b_e1, c_e1, h_e1, w_e1)
        e1 = self.adjust(e1)
        b_e1, c_e1, h_e1, w_e1 = e1.size()
        e1 = e1.view(b_e1, c_e1, h_e1 * w_e1).contiguous().permute(0, 2, 1)
        x8 = self.DecoderBlock4(x8, e1)
        b, hw, c = x8.size()
        h = w = math.isqrt(hw)
        x8 = x8.permute(0, 2, 1).contiguous().view(b, c, h, w)
        x8 = self.upsample_transpose6(x8)

        out = self.output_layer1(x8)
        out = self.output_layer2(out)
        return out

    def param_groups(self, base_lr: float, vit_lr: float):
        """Two LR param groups: everything except the ViT branch at
        `base_lr` (the paper's SGD schedule), and the pretrained ViT at
        a much smaller `vit_lr`. See the note in train_smaformer_baseline.py
        on why a single 0.01 LR across a from-scratch U-Net AND an
        ImageNet-pretrained ViT is a bad idea."""
        vit_params = [p for p in self.vit_branch.vit.parameters() if p.requires_grad]
        other_params = [p for n, p in self.named_parameters()
                        if p.requires_grad and not n.startswith("vit_branch.vit.")]
        return [
            {"params": other_params, "lr": base_lr},
            {"params": vit_params, "lr": vit_lr},
        ]
