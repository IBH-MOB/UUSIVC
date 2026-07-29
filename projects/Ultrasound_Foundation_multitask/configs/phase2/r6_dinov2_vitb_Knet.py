"""R2 — DINOv2 ViT-B/14 (self-supervised) + FPN + Mask2Former.

DINOv2 is the strongest general dense-prediction pretrain (plan2 H1).
ViT outputs a single-scale token map at stride 14; FPN synthesizes the
4-level pyramid Mask2Former expects.

Weights loaded via timm (`vit_base_patch14_dinov2`). The TIMMBackbone
wrapper exposes `out_indices` to tap intermediate transformer blocks;
we tap 4 evenly-spaced layers and feed them to FPN.

Memory note: ViT-B/14 at crop 384 with FPN + Mask2Former (20 queries)
should fit in 24GB at batch 4. If OOM, drop to batch 2 + grad-accum 2.
"""
_base_ = ['./_base_phase2.py']

custom_imports = dict(
    imports=[
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc',
        'projects.Ultrasound_Foundation_multitask.mmseg.evaluation.organs_metric',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.multitask_encoder_decoder',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.cls_head',
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.transforms',
    ])

work_dir = '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/phase2_r2_dinov2_vitb_Knet'

crop_size = (384, 384)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.157926 * 255, 0.157926 * 255, 0.157926 * 255],
    std=[0.199434 * 255, 0.199434 * 255, 0.199434 * 255],
    bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8,
                 betas=(0.9, 0.999))
num_stages = 3
conv_kernel_size = 1
model = dict(
    type='MultitaskEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='TIMMBackbone',
        model_name='vit_base_patch14_dinov2',
        pretrained=True,           # timm auto-downloads DINOv2 weights
        img_size=384,              # DINOv2 default is 518; resize to our crop
        out_indices=(2, 5, 8, 11),  # tap 4 evenly-spaced transformer blocks
        features_only=True),
    neck=dict(
        type='Feature2Pyramid',
        embed_dim=768,
        rescales=[4, 2, 1, 0.5],
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    decode_head=dict(
            type='IterativeDecodeHead',
            num_stages=num_stages,
            kernel_update_head=[
                dict(
                    type='KernelUpdateHead',
                    num_classes=2,
                    num_ffn_fcs=2,
                    num_heads=8,
                    num_mask_fcs=1,
                    feedforward_channels=2048,
                    in_channels=512,
                    out_channels=512,
                    dropout=0.0,
                    conv_kernel_size=conv_kernel_size,
                    ffn_act_cfg=dict(type='ReLU', inplace=True),
                    with_ffn=True,
                    feat_transform_cfg=dict(
                        conv_cfg=dict(type='Conv2d'), act_cfg=None),
                    kernel_updator_cfg=dict(
                        type='KernelUpdator',
                        in_channels=256,
                        feat_channels=256,
                        out_channels=256,
                        act_cfg=dict(type='ReLU', inplace=True),
                        norm_cfg=dict(type='LN'))) for _ in range(num_stages)
            ],
            kernel_generate_head=dict(
                type='FCNHead',
                in_channels=768,
                in_index=3,
                channels=512,
                num_convs=2,
                concat_input=True,
                dropout_ratio=0.1,
                num_classes=2,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))),
    decode_cls_head=dict(
        type='CLSHead',
        in_channels=768,           # Feature2Pyramid deepest output
        in_index=3,
        channels=512,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True,
                         loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,           # Feature2Pyramid deepest output
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend',
             name=work_dir.split('/')[-1]),
        dict(type='WandbVisBackend',
             init_kwargs=dict(project='UUSIVC-mmseg',
                              name=work_dir.split('/')[-1])),
    ],
    name='visualizer')
