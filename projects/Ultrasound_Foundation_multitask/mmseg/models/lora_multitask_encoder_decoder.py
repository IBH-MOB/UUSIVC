# Copyright (c) OpenMMLab. All rights reserved.
"""Multitask encoder-decoder with LoRA on the backbone.

Dropdown sibling of :class:`MultitaskEncoderDecoder` that supports
low-rank adaptation of the backbone while keeping the full model API
(``loss`` / ``predict`` / ``_forward`` / ``train_step``) and checkpoint
layout unchanged (only ``backbone.*.lora_*`` keys are added).

Why LoRA is applied in ``__init__`` (rather than in ``init_weights``):
mmengine's ``Runner`` builds the optim wrapper *before* calling
``model.init_weights()``, so adapters must already exist (with
``requires_grad=True``) when the optimizer is constructed.  Pretrained
weights are handled as follows:

- ``TIMMBackbone(pretrained=True)`` loads weights at construction time, so
  wrapping afterwards is trivially safe.
- Backbones that load via ``init_cfg`` (e.g. mmseg ``VisionTransformer``
  with ``Pretrained``) load through ``load_state_dict`` keys that name the
  *original* layer name (e.g. ``layers.0.attn.qkv.weight``) which no longer
  matches after wrapping.  A registered load-state-dict *pre-hook* re-maps
  those keys to ``<name>.original_layer.*`` transparently, so warm-starting
  from checkpoints keeps working.

See ``lora_layer.py`` for the ``LoRALinear`` primitive.
"""
import re
from typing import Union

import torch.nn as nn
from mmengine.logging import print_log

from mmseg.registry import MODELS

from .lora_layer import LoRALinear
from .multitask_encoder_decoder import MultitaskEncoderDecoder

# Default block/stage-index extractor: supports timm/mmpretrain ViT
# (``blocks.N`` / ``layers.N``) and Swin-style (``stages.N``) backbones.
_DEFAULT_BLOCK_PATTERN = r'(?:blocks|layers|stages)\.(\d+)'
_DEFAULT_TARGETS = [dict(type='qkv')]


@MODELS.register_module()
class LoRAMultitaskEncoderDecoder(MultitaskEncoderDecoder):
    """``MultitaskEncoderDecoder`` with LoRA adaptation of the backbone.

    Accepts every argument of ``MultitaskEncoderDecoder`` plus:

    Args:
        lora_cfg (dict): LoRA configuration with keys
            - ``rank`` (int): LoRA rank. Default 16.
            - ``alpha`` (int): scaling numerator (``scaling = alpha/rank``).
                Default 16.
            - ``drop_rate`` (float): dropout on the LoRA input. Default 0.
            - ``block_indices`` (Sequence[int] | Sequence[tuple]): which
                transformer block / stage indices to patch. ``None`` (default)
                patches all. Integers for single-level indices (ViT
                ``layers``/``timm`` blocks, Swin stages); tuples are matched
                against a two-group ``block_pattern`` (e.g. stage,block).
            - ``block_pattern`` (str): regex whose capture group(s) extract
                the block/stage index from a module name.
                Default ``(?:blocks|layers|stages)\\.(\\d+)``.
            - ``targets`` (list[dict]): per-layer matchers; each dict has
                ``type`` (regex or name suffix), and optional ``rank``,
                ``alpha``, ``drop_rate`` overrides. Default ``[dict(type='qkv')]``.
            - ``save_only_lora`` (bool): when True, only the ``lora_*``
              parameters are written to checkpoints. Default False (full
              state dicts, fully backward-compatible with the base class).
        freeze_backbone (bool): when True all backbone params except the LoRA
            adapters are frozen (``requires_grad=False``); necks/heads remain
            trainable. Default False.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 decode_cls_head,
                 neck=None,
                 neck_cls=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None,
                 lora_cfg: Union[dict, None] = None,
                 freeze_backbone: bool = False):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            decode_cls_head=decode_cls_head,
            neck=neck,
            neck_cls=neck_cls,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.lora_cfg = lora_cfg if lora_cfg is not None else dict()
        self.freeze_backbone = freeze_backbone
        self._lora_applied = False
        self._lora_key_map = {}
        self._patched_names = []

        self._apply_lora()
        self._set_backbone_trainable()
        self._register_lora_hooks()

    # ------------------------------------------------------------------ #
    # LoRA application
    # ------------------------------------------------------------------ #
    def _apply_lora(self) -> None:
        """Wrap target ``nn.Linear`` layers of the backbone with
        ``LoRALinear`` in-place."""
        if self._lora_applied:
            return
        cfg = self.lora_cfg
        block_indices = cfg.get('block_indices', None)
        block_pattern = cfg.get('block_pattern', _DEFAULT_BLOCK_PATTERN)
        targets = cfg.get('targets', None) or _DEFAULT_TARGETS
        default_rank = cfg.get('rank', 16)
        default_alpha = cfg.get('alpha', 16)
        default_drop_rate = cfg.get('drop_rate', 0.0)

        for module_name, module in self.backbone.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            m = re.search(block_pattern, module_name)
            if m is None:
                continue
            groups = tuple(int(g) for g in m.groups())
            if not self._match_block_index(groups, block_indices):
                continue
            target = self._find_target(module_name, targets)
            if target is None:
                continue

            rank = target.get('rank', default_rank)
            alpha = target.get('alpha', default_alpha)
            drop_rate = target.get('drop_rate', default_drop_rate)

            parent_name, leaf = module_name.rsplit('.', 1)
            parent = (self.backbone.get_submodule(parent_name)
                      if parent_name else self.backbone)
            setattr(parent, leaf, LoRALinear(
                module, alpha=alpha, rank=rank, drop_rate=drop_rate))

            self._patched_names.append(module_name)
            self._lora_key_map.update({
                f'{module_name}.weight':
                f'{module_name}.original_layer.weight',
            })
            if module.bias is not None:
                self._lora_key_map[f'{module_name}.bias'] = \
                    f'{module_name}.original_layer.bias'
            print_log(
                f'LoRA applied to backbone.{module_name} '
                f'(rank={rank}, alpha={alpha}, drop_rate={drop_rate})',
                logger='current')

        if not self._patched_names:
            raise ValueError(
                'No backbone layer matched `lora_cfg`. Check `block_indices`, '
                f'`block_pattern` ({block_pattern}) and `targets` ({targets}).')
        self._lora_applied = True

    @staticmethod
    def _match_block_index(groups: tuple, block_indices) -> bool:
        """Match extracted index groups against ``block_indices``."""
        if block_indices is None:
            return True
        if all(isinstance(i, (tuple, list)) for i in block_indices):
            return any(groups == tuple(i) for i in block_indices)
        return len(groups) == 1 and groups[0] in block_indices

    @staticmethod
    def _find_target(module_name: str, targets) -> Union[dict, None]:
        """Return the first target matcher that hits ``module_name``."""
        for target in targets:
            pattern = target['type']
            if re.fullmatch(pattern, module_name) or \
                    module_name.endswith(pattern):
                return target
        return None

    # ------------------------------------------------------------------ #
    # Freeze / trainability
    # ------------------------------------------------------------------ #
    def _set_backbone_trainable(self) -> None:
        """Freeze backbone params (except LoRA adapters) if requested."""
        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Set training mode, keeping frozen norm layers in eval mode."""
        super().train(mode)
        if self.freeze_backbone:
            for module in self.backbone.modules():
                if isinstance(
                        module,
                        (nn.BatchNorm2d, nn.SyncBatchNorm, nn.InstanceNorm2d,
                         nn.LayerNorm)):
                    if not any(p.requires_grad
                               for p in module.parameters()):
                        module.eval()

    # ------------------------------------------------------------------ #
    # State-dict hooks
    # ------------------------------------------------------------------ #
    def _register_lora_hooks(self) -> None:
        """Register key-remap (load) and optional lora-only (save) hooks."""
        # Pre-load hook on the backbone: remap original layer keys that were
        # replaced by LoRALinear wrappers to their ``original_layer.*`` names.
        key_map = self._lora_key_map

        def _remap_keys(state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs):
            remapped = {}
            for key, value in state_dict.items():
                local = key[len(prefix):] if prefix else key
                remapped[prefix + key_map.get(local, local)] = value
            state_dict.clear()
            state_dict.update(remapped)

        self.backbone._register_load_state_dict_pre_hook(_remap_keys)

        # Post-load hook: tolerate lora-only checkpoints in both directions.
        # - full/old ckpt -> lora model: `lora_*` missing keys are dropped
        #   (adapters stay zero-init = standard LoRA warm start)
        # - lora-only ckpt -> lora model: non-lora missing keys are dropped
        def _filter_incompatible(module, incompatible_keys):
            for key in incompatible_keys.missing_keys.copy():
                incompatible_keys.missing_keys.remove(key)
            for key in incompatible_keys.unexpected_keys.copy():
                incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(_filter_incompatible)

        if self.lora_cfg.get('save_only_lora', False):
            def _save_only_lora(module, state_dict, prefix, local_metadata):
                for key in [k for k in state_dict.keys()]:
                    if '.lora_' not in key and not key.endswith('.scaling'):
                        state_dict.pop(key)

            self._register_state_dict_hook(_save_only_lora)

    @property
    def patched_names(self):
        """Names of the backbone modules that received LoRA adapters."""
        return tuple(self._patched_names)

    @property
    def num_lora_parameters(self) -> int:
        """Total number of LoRA parameters (elements, down+up) across the
        backbone."""
        count = 0
        for name, param in self.backbone.named_parameters():
            if 'lora_' in name:
                count += param.numel()
        return count

    @property
    def num_lora_layers(self) -> int:
        """Number of backbone layers that received a LoRA adapter."""
        return len(self._patched_names)