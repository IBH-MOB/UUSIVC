_base_ = [
    '../_base_/datasets/uusivc_dataset.py',
    '../../../../configs/_base_/default_runtime.py',
]

crop_size = (518, 518)
batch_size = 4
# E6 dataset z-score normalization (grayscale ultrasound, replicated to 3ch)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.157926 * 255, 0.157926 * 255, 0.157926 * 255],
    std=[0.199434 * 255, 0.199434 * 255, 0.199434 * 255],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

norm_cfg = dict(type='SyncBN', requires_grad=True)

optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))

# train/test pipelines (E6: PMD + flip, no rotate/clahe/cutout/gamma)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegClsInputs'),
]

train_dataloader = dict(
    batch_size=batch_size, num_workers=2, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)},
                       norm_decay_mult=0.0))

param_scheduler = [
    dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=160000,
         by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=16000,
        save_best='Challenge/Overall',
        rule='greater',
        max_keep_ckpts=3,
    ),
)

auto_scale_lr = dict(enable=False, base_batch_size=batch_size)
