"""Shared Phase 4 fine-tuning settings.

Each child config has its own work directory and changes one major factor.
The checkpoint is loaded as a warm start; the optimizer is intentionally not
resumed from the historical run.
"""

_base_ = ['../phase2/r2_dinov2_vitb_mask2former.py']

data_root_train = '/scratch/dr/UUSIVC26/mmseg_format_full/train'
data_root_val = '/scratch/dr/UUSIVC26/mmseg_format_full/val'
crop_size = (518, 518)
batch_size = 16

load_from = (
    '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs_h200/'
    'phase2_r2_dinov2_vitb_mask2former_batch16_new_metrics/'
    'best_Challenge_Overall_iter_112000.pth')
resume = False

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[40.27113, 40.27113, 40.27113],
    std=[50.85567, 50.85567, 50.85567],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

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
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True, seed=42),
    dataset=dict(
        data_root=data_root_train,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, seed=42),
    dataset=dict(
        data_root=data_root_val,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, seed=42),
    dataset=dict(
        data_root=data_root_val,
        pipeline=test_pipeline))

val_evaluator = dict(data_root=data_root_val)
test_evaluator = dict(data_root=data_root_val)

optimizer = dict(
    type='AdamW',
    lr=1e-5,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1)),
        norm_decay_mult=0.0))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=16000,
        by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=16000,
    val_interval=4000)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='Challenge/Overall',
        rule='greater',
        max_keep_ckpts=4))

auto_scale_lr = dict(enable=False, base_batch_size=batch_size)
