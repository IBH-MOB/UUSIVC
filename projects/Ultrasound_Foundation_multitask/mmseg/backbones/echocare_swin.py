"""EchoCare Swin Transformer backbone for mmsegmentation.

This backbone is a bit-exact reimplementation of the MONAI
``SwinTransformer`` (``use_v2=True``) encoder used by EchoCare, built on top
of mmsegmentation's Swin building blocks (``SwinBlock``, ``SwinBlockSequence``,
``PatchEmbed``).  It loads the exact same pretrained weights
(``echocare_encoder.pth``) after a one-time key-remapping conversion.

Architectural differences from the stock mmseg ``SwinTransformer`` that are
required for weight compatibility and numerical equivalence with the MONAI
encoder:

1. **5 output levels** (not 4): the patch-embedding level (1/2 res) is
   exposed as output 0, and a ``PatchMerging`` downsample is added *after*
   the last stage (1024 -> 2048, 1/32 res) to produce output 4.
2. **``use_v2`` residual conv blocks**: an ``UnetrBasicBlock``-style residual
   3x3 conv (``V2ResBlock``) is inserted before every stage, holding the
   ``layers{N}c`` weights from the MONAI checkpoint.
3. **MONAI-exact ``PatchMerging``**: uses the same ``torch.cat`` strided-slice
   ordering ([TL, BL, TR, BR]) as MONAI's ``PatchMergingV2`` — *not*
   ``nn.Unfold`` like mmseg's stock ``PatchMerging`` (which gives
   [TL, TR, BL, BR]).
4. **Stateless ``proj_out``**: each output is passed through a per-channel
   ``F.layer_norm`` with **no learnable parameters**, matching MONAI's
   ``proj_out(normalize=True)``.  The stock mmseg backbone uses learned
   ``norm0..3`` LayerNorms instead.

With these adaptations the backbone loads the converted EchoCare checkpoint
with ``strict=True`` and produces outputs that are numerically identical
(``torch.allclose``) to the MONAI encoder for the same input.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS
from mmseg.models.backbones.swin import SwinBlockSequence
from mmseg.models.utils.embed import PatchEmbed


class EchoCarePatchMerging(BaseModule):
    """Patch merging that bit-exactly matches MONAI's ``PatchMergingV2``.

    MONAI groups 2x2 neighbourhoods via ``torch.cat([x[:, j::2, i::2, :] for
    i, j in product(2, 2)], -1)`` yielding the order [TL, BL, TR, BR].  The
    stock mmseg ``PatchMerging`` uses ``nn.Unfold`` which yields [TL, TR, BL,
    BR] — a different column ordering in the ``reduction`` Linear input, so
    the same weights cannot be loaded.  This class reproduces MONAI's exact
    cat-based forward so the ``reduction`` / ``norm`` weights transfer
    verbatim.

    Args:
        dim: input channel count (the stage's embed dim).
        out_channels: output channel count (2 * dim for Swin).
        norm_cfg: config for the LayerNorm over the concatenated 4*dim dim.
        init_cfg: init config.
    """

    def __init__(self,
                 dim: int,
                 out_channels: int,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.dim = dim
        self.in_channels = dim
        self.out_channels = out_channels
        sample_dim = 4 * dim
        self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        self.reduction = nn.Linear(sample_dim, out_channels, bias=False)

    def forward(self, x, input_size):
        """Forward.

        Args:
            x: (B, H*W, C) flattened token sequence.
            input_size: (H, W) spatial shape.

        Returns:
            x: (B, H/2*W/2, out_channels)
            output_size: (H/2, W/2)
        """
        B, L, C = x.shape
        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C)

        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        x = torch.cat(
            [x[:, j::2, i::2, :] for i, j in itertools.product(range(2),
                                                               range(2))],
            dim=-1,
        )
        x = self.norm(x)
        x = self.reduction(x)

        out_h, out_w = (H + pad_h) // 2, (W + pad_w) // 2
        x = x.view(B, out_h * out_w, self.out_channels)
        return x, (out_h, out_w)


class V2ResBlock(nn.Module):
    """Residual conv block replicating MONAI's ``UnetrBasicBlock`` (
    ``res_block=True``) / ``UnetResBlock``.

    For EchoCare, ``in_channels == out_channels`` and ``stride == 1``, so no
    ``conv3`` / ``norm3`` projection shortcut is created.  The only learnable
    parameters are ``conv1.weight`` and ``conv2.weight`` (both bias-free 3x3
    convs).  ``InstanceNorm2d`` is affine-free (no params) and ``LeakyReLU``
    uses ``negative_slope=0.01`` (inplace), matching MONAI's defaults.

    Forward (matching ``UnetResBlock.forward``)::

        out = lrelu(norm1(conv1(x)))
        out = norm2(conv2(out))
        out = lrelu(out + x)
    """

    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size, stride, pad, bias=False)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size, 1, pad, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.lrelu(out)
        return out


@MODELS.register_module()
class EchoCareSwinTransformer(BaseModule):
    """EchoCare Swin Transformer backbone (MONAI SwinUNETR v2 encoder).

    Reuses mmseg's ``PatchEmbed``, ``SwinBlockSequence`` (which contains
    ``SwinBlock`` / ``ShiftWindowMSA`` / ``WindowMSA``) and adds the
    MONAI-specific pieces (``EchoCarePatchMerging``, ``V2ResBlock``,
    stateless ``proj_out``, 5th downsample) to be bit-exact with the MONAI
    ``SwinTransformer(use_v2=True)`` encoder.

    Args:
        in_channels: input image channels (3 for EchoCare).
        embed_dims: base embedding dim (128 for EchoCare).
        patch_size: patch size (2 for EchoCare).
        window_size: window size **as int** (8 for EchoCare).  Unlike the
            stock mmseg Swin, this is always converted to a 2-tuple
            internally; pass an int.
        depths: per-stage block counts.
        num_heads: per-stage head counts.
        mlp_ratio: MLP hidden dim ratio.
        qkv_bias: QKV bias flag.
        use_v2: insert residual conv blocks before each stage (EchoCare
            uses True).
        out_indices: which output levels to return (0-4).  Level 0 is the
            patch-embedding output; levels 1-4 are post-downsample stage
            outputs.
        patch_norm: add LayerNorm after patch embedding (EchoCare uses
            False — no norm, matching MONAI).
        strides: per-stage stride for PatchMerging (all 2 for EchoCare;
            strides[0] must equal patch_size).
        drop_rate: dropout rate (0 for EchoCare).
        attn_drop_rate: attention dropout rate (0 for EchoCare).
        drop_path_rate: stochastic depth rate (0 for EchoCare).
        with_cp: use gradient checkpointing.
        act_cfg: activation config (GELU).
        norm_cfg: norm config (LayerNorm) for block norms and patch merging.
        pretrained: legacy pretrained path (converted to init_cfg).
        init_cfg: init config, e.g. ``dict(type='Pretrained',
            checkpoint='...')``.
    """

    def __init__(self,
                 in_channels: int = 3,
                 embed_dims: int = 128,
                 patch_size: int = 2,
                 window_size: int = 8,
                 depths: Sequence[int] = (2, 2, 18, 2),
                 num_heads: Sequence[int] = (4, 8, 16, 32),
                 mlp_ratio: float = 4,
                 qkv_bias: bool = True,
                 use_v2: bool = True,
                 out_indices: Sequence[int] = (0, 1, 2, 3, 4),
                 patch_norm: bool = False,
                 strides: Sequence[int] = (2, 2, 2, 2),
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 with_cp: bool = False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 pretrained=None,
                 init_cfg=None):
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'
        assert len(depths) == len(num_heads) == len(strides)
        max_out = len(depths)  # 4 stages -> max output level = 4
        for idx in out_indices:
            assert 0 <= idx <= max_out, \
                f'out_indices must be in [0, {max_out}], got {idx}'

        self.out_indices = tuple(out_indices)
        self.use_v2 = use_v2
        self.num_layers = len(depths)
        if isinstance(window_size, (list, tuple)):
            assert len(window_size) == 2 and window_size[0] == window_size[1], \
                'window_size must be an int or a 2-element uniform tuple/list'
            window_size = window_size[0]

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.v2_blocks: ModuleList = ModuleList()
        self.stages: ModuleList = ModuleList()
        in_ch = embed_dims
        for i in range(self.num_layers):
            if use_v2:
                self.v2_blocks.append(
                    V2ResBlock(channels=in_ch, kernel_size=3, stride=1))

            downsample = EchoCarePatchMerging(
                dim=in_ch,
                out_channels=2 * in_ch,
                norm_cfg=norm_cfg,
                init_cfg=None)

            stage = SwinBlockSequence(
                embed_dims=in_ch,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_ch),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            in_ch = 2 * in_ch

        self.num_features = [int(embed_dims * 2**i)
                             for i in range(self.num_layers)]
        self.embed_dims = embed_dims

    def proj_out(self, x):
        """Stateless per-channel LayerNorm, matching MONAI's
        ``proj_out(normalize=True)``.

        Args:
            x: (B, C, H, W) NCHW tensor.

        Returns:
            (B, C, H, W) with ``F.layer_norm`` applied over the channel dim
            (no learnable weight/bias, eps=1e-5).
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, [C])
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _tokens_to_nchw(self, x, hw_shape):
        """(B, N, C) + (H, W) -> (B, C, H, W)."""
        B, N, C = x.shape
        H, W = hw_shape
        assert N == H * W
        return x.transpose(1, 2).contiguous().view(B, C, H, W)

    def _nhwc_to_tokens(self, x):
        """(B, C, H, W) -> (B, H*W, C)."""
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2).contiguous()

    def init_weights(self):
        if self.init_cfg is None:
            print_log(
                f'No pre-trained weights for {self.__class__.__name__}, '
                f'training from scratch', logger='current')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
        else:
            assert 'checkpoint' in self.init_cfg, \
                f'Only support Pretrained init_cfg in {self.__class__.__name__}'
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif isinstance(ckpt, dict) and 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                elif k.startswith('module.'):
                    state_dict[k[7:]] = v
                else:
                    state_dict[k] = v

            missing, unexpected = self.load_state_dict(
                state_dict, strict=False)
            if missing:
                print_log(
                    f'Missing keys ({len(missing)}): {missing[:10]}'
                    f'{"..." if len(missing) > 10 else ""}', logger='current')
            if unexpected:
                print_log(
                    f'Unexpected keys ({len(unexpected)}): {unexpected[:10]}'
                    f'{"..." if len(unexpected) > 10 else ""}',
                    logger='current')
            if not missing and not unexpected:
                print_log(
                    f'Successfully loaded all weights for '
                    f'{self.__class__.__name__} from '
                    f'{self.init_cfg["checkpoint"]}', logger='current')

    def forward(self, x):
        """Forward.

        Args:
            x: (B, 3, H, W) input image.

        Returns:
            list of output feature maps.  With ``out_indices=(0,1,2,3,4)``,
            returns 5 tensors:
            - level 0: (B, 128, H/2, W/2)
            - level 1: (B, 256, H/4, W/4)
            - level 2: (B, 512, H/8, W/8)
            - level 3: (B, 1024, H/16, W/16)
            - level 4: (B, 2048, H/32, W/32)
        """
        x, hw_shape = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self._tokens_to_nchw(x, hw_shape)

        outs = []
        if 0 in self.out_indices:
            outs.append(self.proj_out(x))

        for i in range(self.num_layers):
            if self.use_v2:
                x = self.v2_blocks[i](x)

            x_tokens = self._nhwc_to_tokens(x)
            x_tokens, hw_shape, _, _ = self.stages[i](x_tokens, hw_shape)
            x = self._tokens_to_nchw(x_tokens, hw_shape)

            level = i + 1
            if level in self.out_indices:
                outs.append(self.proj_out(x))
        return outs
