_base_ = ['./_base_dino.py']

custom_imports = dict(
    imports=[
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc',
        'projects.Ultrasound_Foundation_multitask.mmseg.evaluation.organs_metric',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.multitask_encoder_decoder',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.cls_head',
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.transforms',
    ])

work_dir = './work_dirs/dinov3_vit_small_Knet_320k'

crop_size = (256, 256)
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
        model_name='vit_small_patch16_dinov3',
        pretrained=True,           # timm auto-downloads DINOv3 weights
        img_size=256,              
        out_indices=(2, 5, 8, 11),  # tap 4 evenly-spaced transformer blocks
        features_only=True),
    neck=dict(
        type='Feature2Pyramid',
        embed_dim=384,
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
            in_channels=384,
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
        in_channels=384,           # Feature2Pyramid deepest output
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
        in_channels=384,           # Feature2Pyramid deepest output
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

# ViT has no Swin-style stages. Generic {'backbone': lr_mult=0.1} from
# _base_phase2 covers the whole backbone. The FPN neck uses full LR.

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
