# Copyright (c) OpenMMLab. All rights reserved.
"""LoRA primitives for backbone adaptation.

A self-contained port of mmpretrain's ``LoRALinear``
(``mmpretrain/models/peft/lora.py``), kept as a plain ``nn.Module`` so it can
be dropped in anywhere without registering anything in the registries.

Forward math::

    y = W0 x + (lora_up(lora_down(dropout(x)))) * (alpha / rank)

The original layer is stored (and called) as-is; only the small ``lora_down`` /
``lora_up`` matrices are trained, so the wrapped base weights can be kept fully
frozen.
"""
import itertools
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` with a learnable low-rank adapter.

    Args:
        original_layer (nn.Linear): The linear layer to be fine-tuned.
        alpha (int): Scale factor of LoRA. Defaults to 1.
        rank (int): Rank of LoRA. Defaults to 0.
        drop_rate (float): Dropout rate for the LoRA input. Defaults to 0.
    """

    def __init__(self,
                 original_layer: nn.Linear,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.):
        super().__init__()
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        # Registered as a persistent buffer so the merge tool can read the
        # scaling factor straight from a saved checkpoint.
        self.register_buffer('scaling', torch.tensor(alpha / rank),
                             persistent=True)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.original_layer = original_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original_layer(x)

        lora_x = self.lora_dropout(x)
        lora_out = self.lora_up(self.lora_down(lora_x)) * self.scaling

        return out + lora_out

    def merge(self) -> nn.Linear:
        """Fold ``lora_down``/``lora_up`` into the original layer.

        Returns:
            nn.Linear: ``self.original_layer`` in-place updated to
            ``W0 + (B @ A) * (alpha / rank)`` and with the LoRA adapter
            removed (module returned is no longer a component of this
            ``LoRALinear``).
        """
        with torch.no_grad():
            merged = self.original_layer.weight + (
                (self.lora_up.weight @ self.lora_down.weight) * self.scaling)
            self.original_layer.weight.copy_(merged)
        return self.original_layer

    @property
    def lora_parameters(self):
        """Iterable over the learnable LoRA parameters (down + up)."""
        return itertools.chain(self.lora_down.parameters(),
                               self.lora_up.parameters())


def is_lora_trainable(module) -> bool:
    """``True`` if ``module`` is a LoRA layer with rank > 0."""
    return isinstance(module, LoRALinear) and module.lora_down.out_features > 0
