#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Merge LoRA adapters back into the base weights of a
``LoRAMultitaskEncoderDecoder`` checkpoint.

Produces a standard full checkpoint (same layout as a plain
``MultitaskEncoderDecoder`` run) that can be loaded directly by the
*unmodified* configs (e.g. ``configs/phase2/r2_dinov2_vitb_mask2former.py``)
for inference / deployment.

Math per adapted layer::

    W_new = W0 + (lora_up @ lora_down) * scaling          (scaling = alpha/rank)

Key-format assumption (project wrapper layout)::

    backbone.timm_model.blocks.8.attn.qkv.original_layer.weight
    backbone.timm_model.blocks.8.attn.qkv.lora_down.weight
    backbone.timm_model.blocks.8.attn.qkv.lora_up.weight
    backbone.timm_model.blocks.8.attn.qkv.scaling        (buffer)

        merged -> backbone.timm_model.blocks.8.attn.qkv.weight
                  backbone.timm_model.blocks.8.attn.qkv.bias

Usage::

    python tools/merge_lora_weight.py <src_lora_ckpt.pth> <dst_merged.pth>
"""
import argparse
from pathlib import Path

import torch

LORA_SUFFIXES = ('lora_down', 'lora_up')


def _collect(ckpt: dict):
    """Group state-dict entries into: plain, lora pairs, scaling, originals.

    Returns:
        tuple: (plain_state_dict, merged_entries) where ``merged_entries`` is
        a dict ``prefix -> dict(weight=..., bias=..., down=..., up=...,
        scaling=...)``.
    """
    state_dict = ckpt['state_dict']
    plain = {}
    merged_entries = {}
    for name, param in state_dict.items():
        if name.endswith('.scaling'):
            prefix = name[:-len('.scaling')]
            merged_entries.setdefault(prefix, {})['scaling'] = param
            continue
        parts = name.split('.')
        if len(parts) >= 2 and parts[-2] in LORA_SUFFIXES:
            prefix = '.'.join(parts[:-2])
            merged_entries.setdefault(
                prefix, {}).setdefault(parts[-2], param)
        elif name.endswith('.original_layer.weight'):
            prefix = name[: -len('.original_layer.weight')]
            merged_entries.setdefault(prefix, {})['weight'] = param
        elif name.endswith('.original_layer.bias'):
            prefix = name[: -len('.original_layer.bias')]
            merged_entries.setdefault(prefix, {})['bias'] = param
        else:
            plain[name] = param
    return plain, merged_entries


def merge_lora_weight(ckpt: dict) -> dict:
    """Merge every LoRA pair in the checkpoint into the base weights."""
    plain, entries = _collect(ckpt)
    merged_ckpt = {'state_dict': plain, 'meta': ckpt.get('meta', {})}

    for prefix, entry in entries.items():
        if 'lora_down' not in entry or 'lora_up' not in entry:
            raise ValueError(
                f'Incomplete LoRA pair for `{prefix}`: '
                f'found {sorted(entry)}. The checkpoint may already be merged '
                'or use a different key layout.')
        if 'weight' not in entry:
            raise ValueError(
                f'No original weight found for `{prefix}`; the checkpoint '
                'does not look like a LoRAMultitaskEncoderDecoder save.')
        scaling = entry.get('scaling')
        if scaling is None:
            raise ValueError(
                f'Missing `scaling` buffer for `{prefix}`; cannot determine '
                'alpha/rank. Re-save with the project wrapper (scaling is a '
                'persistent buffer).')
        down, up, w0 = entry['lora_down'], entry['lora_up'], entry['weight']
        merged = w0 + (up @ down) * scaling
        plain[f'{prefix}.weight'] = merged
        if 'bias' in entry:
            plain[f'{prefix}.bias'] = entry['bias']

    return merged_ckpt


def main():
    parser = argparse.ArgumentParser(description='Merge LoRA weights')
    parser.add_argument('src', help='LoRA checkpoint path (mmengine format)')
    parser.add_argument('dst', help='destination merged checkpoint path')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    assert src.exists(), f'checkpoint not found: {src}'
    dst.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(src, map_location='cpu')
    if 'state_dict' not in ckpt:
        raise ValueError(
            f'{src} is not an mmengine checkpoint '
            "(missing 'state_dict' key).")
    merged = merge_lora_weight(ckpt)
    torch.save(merged, dst)
    print(f'[merge_lora_weight] wrote {dst} '
          f'({len(merged["state_dict"])} keys)')


if __name__ == '__main__':
    main()
