# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.utils import ConfigType, SampleList
from mmseg.models.utils import resize
from mmseg.models.losses import accuracy
from typing import Tuple
from torch import Tensor
from typing import List, Tuple
from mmpretrain.structures import DataSample
import torch.nn.functional as F

@MODELS.register_module()
class CLSHeadM2F(BaseDecodeHead):
    """Fully Convolution Networks for classification.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    with MLP in the end

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_layers=2,
                 concat_input=True,
                 pool_type='mean',
                 dropout_ratio=0.1,
                 **kwargs):
        assert num_layers >= 0
        assert pool_type in ('mean', 'max', 'attn')
        self.num_layers = num_layers
        self.concat_input = concat_input
        self.pool_type = pool_type
        super().__init__(input_transform=None, **kwargs)

        if num_layers == 0:
            assert self.in_channels == self.channels
            self.mlp = nn.Identity()
        else:
            layers = []
            in_ch = self.in_channels
            for _ in range(num_layers):
                layers += [
                    nn.Linear(in_ch, self.channels),
                    nn.LayerNorm(self.channels),
                    nn.ReLU(inplace=True),
                ]
                in_ch = self.channels
            self.mlp = nn.Sequential(*layers)

        if self.concat_input:
            self.proj_cat = nn.Sequential(
                nn.Linear(self.in_channels + self.channels, self.channels),
                nn.LayerNorm(self.channels),
                nn.ReLU(inplace=True))

        if self.pool_type == 'attn':
            self.attn_pool = nn.Linear(self.channels, 1)

        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 \
            else nn.Identity()

        self.num_classes = kwargs.get('num_classes', 2)
        self.fc = nn.Linear(self.channels, self.num_classes)

        self.ignore_index = 2 # ignore instances with ground truth label 2 

    def _transform_inputs(self, inputs):
        """Accepts either a single query tensor (B, N, C), or a list of
        per-decoder-layer query tensors (in which case the last layer's
        queries are used)."""
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[-1]
        return inputs

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.mlp(x)
        if self.concat_input:
            feats = self.proj_cat(torch.cat([x, feats], dim=-1))
        return feats

    def _pool(self, feats):
        """Aggregate per-query features into one vector per image.

        Args:
            feats (Tensor): (batch_size, num_queries, channels)

        Returns:
            Tensor: (batch_size, channels)
        """
        if self.pool_type == 'mean':
            return feats.mean(dim=1)
        elif self.pool_type == 'max':
            return feats.max(dim=1).values
        else:  # attn
            weights = torch.softmax(self.attn_pool(feats), dim=1)  # (B,N,1)
            return (feats * weights).sum(dim=1)
        
    def forward(self, inputs):
        """Forward function."""
        feats = self._forward_feature(inputs)
        pooled = self._pool(feats)
        pooled = self.dropout(pooled)
        output = self.fc(pooled)
        return output

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType = None) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        cls_logits = self.forward(inputs)
        losses = self.loss_by_feat(cls_logits, batch_data_samples)
        return losses
    

    def loss_by_feat(self, cls_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute classification loss.

        Args:
            cls_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        cls_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        # cls_logits = resize(
        #     input=cls_logits,
        #     size=cls_label.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # if self.sampler is not None:
        #     cls_weight = self.sampler.sample(cls_logits, cls_label)
        # else:
        #     cls_weight = None
        # seg_label = cls_label.squeeze(1)

        cls_label = cls_label.to(cls_logits.device)
        cls_weight = None

        if (cls_label == 2).all():
            # print("All elements are equal to 2")
            return {"loss_ce": torch.tensor(0.0, device='cuda:0'), "acc_cls": torch.tensor(0.0, device='cuda:0')}

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    cls_logits,
                    cls_label,
                    weight=cls_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    cls_logits,
                    cls_label,
                    weight=cls_weight,
                    ignore_index=self.ignore_index)

        loss['acc_cls'] = accuracy(
            cls_logits, cls_label, ignore_index=self.ignore_index)
        return loss
    
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_cls = [
            torch.tensor(data_sample.gt_label[0]) for data_sample in batch_data_samples
        ]
        return torch.stack(gt_cls, dim=0)
    

    def predict_by_feat(self, cls_logits: Tensor,
                        batch_img_metas: List[dict]= None) -> Tensor:
        """Transform a batch of output cls_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        # if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
        #     # slide inference
        #     size = batch_img_metas[0]['img_shape']
        # elif 'pad_shape' in batch_img_metas[0]:
        #     size = batch_img_metas[0]['pad_shape'][:2]
        # else:
        #     size = batch_img_metas[0]['img_shape']

        # seg_logits = resize(
        #     input=seg_logits,
        #     size=size,
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return cls_logits

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType = None) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        cls_score = self.forward(inputs)

        cls_score = self._get_predictions(cls_score,batch_img_metas)
        # return self.predict_by_feat(cls_score, batch_img_metas)
        return cls_score
    
    
    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples