"""
ViTBottleneckBranch
====================
Why this exists (read this before you use it)
-----------------------------------------------
The SMAFormer paper states the backbone is "initialized with a ViT
pre-trained model." The released architecture (smaformer_core.py) is
NOT a standard ViT, though: it's a custom conv+attention U-Net where
every stage has a different channel width (16 -> 32 -> 128 -> 512) and
a different spatial resolution, built with a bespoke conv-based patch
embedding (`Modulator.PE`) rather than a single linear patch-embed +
stack of pre-norm transformer blocks at one fixed width. A standard
`timm` ViT-B/16 state_dict (patch_embed.proj, blocks.0..11, pos_embed,
cls_token, ...) has no clean 1:1 mapping onto that structure -- there
is no "load_state_dict(vit_checkpoint, strict=True)" that will work
here without silently discarding most of the checkpoint.

So instead of pretending a clean drop-in load exists, this module
does the closest defensible thing: it runs the full image through a
real pretrained ViT-B/16, takes its patch-token grid, and projects it
into the exact shape SMAFormer's bottleneck expects (target_grid x
target_grid spatial, 512 channels). That projected tensor is then
fused *additively, through a zero-initialized gate* into the SMAFormer
bottleneck (see smaformer_vit_baseline.py). Zero-init means training
starts identical to the vanilla SMAFormer forward pass and the model
has to learn to use the ViT prior -- it can't be hurt by a bad init.

Grid arithmetic (worked example at this project's actual 224x224 input
-- NOTE: an earlier version of this docstring used 512x512 as the
running example, which this project never actually trains at; keeping
that stale number here was exactly the kind of thing that misleads a
debugging session later, so it's been replaced with the real numbers)
-----------------------------------------------
- vit_base_patch16_224 pretrained natively on 224x224 -> 14x14 patch
  grid. Since this project also trains at 224x224 (ViT-B/16's native
  resolution), timm needs zero positional-embedding interpolation --
  a real advantage over training at a different resolution, where
  timm's `img_size` override would bicubic-interpolate the pretrained
  pos_embed and introduce some, usually small, adaptation artifact.
- SMAFormer's bottleneck is img_size/32 per side (5 stride-2 stages:
  three Modulator.PE stages + two stride-2 ResidualConvs) -- at 224
  input that's 7x7. So we strided-conv-project the 14x14xhidden_dim
  ViT grid down to 7x7x512 in one learned step (stride-2 conv). This
  adds a small number of from-scratch parameters, which is unavoidable
  -- the grids don't match natively -- but it costs almost nothing and
  keeps ~86M pretrained ViT-B parameters intact.
- This grid math generalizes to any img_size divisible by 32 (see
  target_grid's default below), not just 224 -- it just happens that
  224 is what this project actually uses.
"""

import timm
import torch
import torch.nn as nn


class ViTBottleneckBranch(nn.Module):
    def __init__(
        self,
        vit_name: str = "vit_base_patch16_224",
        img_size: int = 224,
        target_grid: int | None = None,
        target_ch: int = 512,
        pretrained: bool = True,
        freeze_blocks: int = 0,
    ):
        """
        vit_name:       any timm ViT that exposes `patch_embed.grid_size`
                         and `forward_features` (vit_base_patch16_224,
                         vit_base_patch32_224, deit3_base_patch16_224, ...).
        img_size:       resolution you actually feed the branch. Default
                         224 -- ViT-B/16's native pretrained resolution,
                         which is what this project actually trains at
                         (no positional-embedding interpolation needed).
        target_grid:    spatial grid SMAFormer's bottleneck expects.
                         Defaults to img_size // 32 (SMAFormer's 5
                         stride-2 stages) if not given explicitly -- this
                         used to default to a hardcoded 16 (correct only
                         at img_size=512), which is exactly the kind of
                         mismatch that previously caused a ValueError at
                         construction time when img_size changed but this
                         default silently didn't.
        target_ch:      channel width SMAFormer's bottleneck expects (512,
                         i.e. filters[5] in smaformer_core.SMAFormer).
        pretrained:     download ImageNet weights via timm (needs internet;
                         set False to sanity-check shapes offline).
        freeze_blocks:  number of leading transformer blocks to freeze
                         (0 = fully fine-tune, len(blocks) = fully frozen).
                         Freezing early blocks and fine-tuning only the
                         last few is the usual recipe on a ~2000-image
                         dataset like ISIC 2018, where fully fine-tuning
                         an 86M-parameter ViT end to end tends to overfit.
        """
        super().__init__()
        if target_grid is None:
            target_grid = img_size // 32
        self.img_size = img_size
        self.vit = timm.create_model(
            vit_name, pretrained=pretrained, img_size=img_size, num_classes=0
        )
        self.vit_dim = self.vit.embed_dim
        self.vit_grid = self.vit.patch_embed.grid_size  # e.g. (32, 32)
        self.num_prefix_tokens = getattr(self.vit, "num_prefix_tokens", 1)

        stride_h = self.vit_grid[0] // target_grid
        stride_w = self.vit_grid[1] // target_grid
        if self.vit_grid[0] % target_grid != 0 or self.vit_grid[1] % target_grid != 0:
            raise ValueError(
                f"ViT patch grid {self.vit_grid} is not an integer multiple of "
                f"target_grid={target_grid}. Pick an img_size/patch size combo "
                f"where it divides evenly (512 input + patch16 -> 32x32 grid "
                f"-> divides evenly by 16)."
            )

        self.target_grid = target_grid
        self.target_ch = target_ch
        self.project = nn.Sequential(
            nn.Conv2d(self.vit_dim, target_ch, kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w)),
            nn.BatchNorm2d(target_ch),
            nn.GELU(),
        )

        self.set_frozen_blocks(freeze_blocks)

    def set_frozen_blocks(self, freeze_blocks: int):
        """Freeze the first `freeze_blocks` transformer blocks, plus
        patch_embed/cls_token/pos_embed/norm whenever any freezing is
        requested at all. Call again any time to change the schedule
        (e.g. unfreeze more blocks after warmup).

        freeze_blocks=0 now genuinely means full fine-tune (everything
        trainable, including patch_embed/pos_embed/cls_token/norm) and
        freeze_blocks=len(blocks) genuinely means fully frozen (norm
        included) -- previously patch_embed/cls_token/pos_embed were
        frozen unconditionally regardless of freeze_blocks, and norm was
        never frozen at all, so neither documented extreme actually did
        what its docstring claimed.
        """
        embed_requires_grad = freeze_blocks == 0
        for p in self.vit.patch_embed.parameters():
            p.requires_grad = embed_requires_grad
        if self.vit.cls_token is not None:
            self.vit.cls_token.requires_grad = embed_requires_grad
        self.vit.pos_embed.requires_grad = embed_requires_grad

        for i, block in enumerate(self.vit.blocks):
            requires_grad = i >= freeze_blocks
            for p in block.parameters():
                p.requires_grad = requires_grad

        if hasattr(self.vit, "norm") and self.vit.norm is not None:
            norm_requires_grad = freeze_blocks < len(self.vit.blocks)
            for p in self.vit.norm.parameters():
                p.requires_grad = norm_requires_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, img_size, img_size] -> [B, target_ch, target_grid, target_grid]"""
        assert x.shape[-2:] == (self.img_size, self.img_size), (
            f"ViTBottleneckBranch was constructed for img_size={self.img_size} but "
            f"received input of shape {tuple(x.shape[-2:])}. timm's PatchEmbed bakes "
            f"a fixed positional-embedding size in at construction time and will "
            f"hard-crash on a mismatch -- this assertion just makes the failure point "
            f"to your data pipeline instead of three layers deep inside timm."
        )
        feats = self.vit.forward_features(x)              # [B, N + prefix, vit_dim]
        feats = feats[:, self.num_prefix_tokens:, :]       # drop cls/register tokens
        b, n, c = feats.shape
        h = w = int(n ** 0.5)
        feats = feats.permute(0, 2, 1).reshape(b, c, h, w)  # [B, vit_dim, h, w]
        feats = self.project(feats)                         # [B, target_ch, target_grid, target_grid]
        return feats
