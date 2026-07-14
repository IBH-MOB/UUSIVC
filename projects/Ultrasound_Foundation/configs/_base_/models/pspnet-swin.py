# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='EchoCareSwinTransformer',
        in_channels=3,
        embed_dims=128,
        window_size=8,
        patch_size=2,
        strides=(2, 2, 2, 2),
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        mlp_ratio=4,
        qkv_bias=True,
        use_v2=True,
        out_indices=(0, 1, 2, 3, 4),
        patch_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/scratch/dr/o.iraqy/UUSIVC-MMSeg/weights/'
                       'echocare_encoder_mmseg.pth')),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=4,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
