# Copyright (c) OpenMMLab. All rights reserved.
"""Organ-conditioned classification head.

Minimal test of per-organ conditioning: a learned ``nn.Embedding`` keyed by
the dataset folder name (``data_sample.metainfo['organ']``) is concatenated
to the GAP-pooled feature before the final linear classifier.  Everything
else (loss, postprocess, routing) is inherited unchanged from :class:`CLSHead`,
so no other experiment is affected.
"""
import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .cls_head import CLSHead


@MODELS.register_module()
class OrganCondCLSHead(CLSHead):
    """``CLSHead`` with a learned per-organ embedding concatenated to the
    pooled feature before the linear classifier.

    Args:
        organ_names (Sequence[str]): Ordered list of organ folder names; the
            position in the list is the embedding index.  Names not found at
            runtime fall back to index 0.
        organ_embed_dim (int): Dimension of the per-organ embedding.  Default 16.
    """

    def __init__(self,
                 organ_names,
                 organ_embed_dim=16,
                 **kwargs):
        super().__init__(**kwargs)
        self.organ_names = list(organ_names)
        self.organ2idx = {name: i for i, name in enumerate(self.organ_names)}
        self.num_organs = len(self.organ_names)
        self.organ_embed_dim = organ_embed_dim
        self.organ_embed = nn.Embedding(self.num_organs, organ_embed_dim)
        self.fc = nn.Linear(self.out_channels + organ_embed_dim,
                            self.num_classes)

    def _organ_idx(self, samples, device):
        idxs = []
        for s in samples:
            meta = s.metainfo if hasattr(s, 'metainfo') else s
            idxs.append(self.organ2idx.get(meta.get('organ'), 0))
        return torch.tensor(idxs, dtype=torch.long, device=device)

    def forward(self, inputs, organ_idx=None):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        output = self.gap(output)
        output = torch.flatten(output, 1)
        if organ_idx is None:
            emb = output.new_zeros(output.size(0), self.organ_embed_dim)
        else:
            emb = self.organ_embed(organ_idx)
        output = torch.cat([output, emb], dim=1)
        output = self.fc(output)
        return output

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        device = inputs[0].device if isinstance(inputs, (list, tuple)) \
            else inputs.device
        organ_idx = self._organ_idx(batch_data_samples, device)
        cls_logits = self.forward(inputs, organ_idx)
        return self.loss_by_feat(cls_logits, batch_data_samples)

    def predict(self, inputs, batch_img_metas, test_cfg=None):
        device = inputs[0].device if isinstance(inputs, (list, tuple)) \
            else inputs.device
        organ_idx = self._organ_idx(batch_img_metas, device)
        cls_score = self.forward(inputs, organ_idx)
        return self._get_predictions(cls_score, batch_img_metas)
