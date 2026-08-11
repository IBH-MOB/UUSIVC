_base_ = ['./_base_dino.py']

custom_imports = dict(
    imports=[
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc',
        'projects.Ultrasound_Foundation_multitask.mmseg.evaluation.organs_metric',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.multitask_encoder_decoder',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.cls_head',
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.transforms',
    ])

work_dir = './work_dirs/dinov3_vit_small_mask2former_b16_512x512'

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.157926 * 255, 0.157926 * 255, 0.157926 * 255],
    std=[0.199434 * 255, 0.199434 * 255, 0.199434 * 255],
    bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8,
                 betas=(0.9, 0.999))

model = dict(
    type='MultitaskEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='TIMMBackbone',
        model_name='vit_small_patch16_dinov3',
        pretrained=True,           # timm auto-downloads DINOv3 weights
        img_size=512,              
        out_indices=(2, 5, 8, 11),  # tap 4 evenly-spaced transformer blocks
        features_only=True),
    neck=dict(
        type='Feature2Pyramid',
        embed_dim=384,
        rescales=[4, 2, 1, 0.5],
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[384, 384, 384, 384],   # Feature2Pyramid output (embed_dim)
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=2,
        num_queries=20,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            strides=[4, 8, 16, 32],
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256, num_heads=8, num_levels=3,
                        num_points=4, im2col_step=64, dropout=0.0,
                        batch_first=True, norm_cfg=None, init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=1024,
                        num_fcs=2, ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0,
                    dropout_layer=None, batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0,
                    dropout_layer=None, batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0,
                    dropout_layer=None, add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0,
            reduction='mean', class_weight=[1.0] * 2 + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss', use_sigmoid=True, activate=True,
            reduction='mean', naive_dice=True, eps=1.0, loss_weight=5.0),
        train_cfg=dict(
            num_points=12544, oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(type='mmdet.CrossEntropyLossCost', weight=5.0,
                         use_sigmoid=True),
                    dict(type='mmdet.DiceCost', weight=5.0, pred_act=True,
                         eps=1.0)]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
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
