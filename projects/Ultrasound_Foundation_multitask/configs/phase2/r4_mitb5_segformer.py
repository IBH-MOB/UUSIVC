"""R4 — SegFormer MiT-B5 + SegFormerHead.

Lightweight decoder test (plan2 H3). MiT-B5 outputs 4 levels at strides
[4, 8, 16, 32] with channels [64, 128, 320, 512]. SegFormerHead is all-MLP,
much lighter than Mask2Former. If it matches Mask2Former within noise,
prefer it for faster iteration.

NOTE: SegFormerHead is a per-pixel classifier; it does not use the
Hungarian matching / query mechanism. Auxiliary FCN head is kept for
deep supervision. CLSHead taps the deepest MiT level (512 ch, index 3).
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

work_dir = '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/phase2_r4_mitb5_segformer'

crop_size = (384, 384)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.157926 * 255, 0.157926 * 255, 0.157926 * 255],
    std=[0.199434 * 255, 0.199434 * 255, 0.199434 * 255],
    bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8,
                 betas=(0.9, 0.999))

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa

model = dict(
    type='MultitaskEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        mlp_ratio=4,
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        interpolate_mode='bilinear',
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0)),
    decode_cls_head=dict(
        type='CLSHead',
        in_channels=512,
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
        in_channels=320,
        in_index=2,
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
