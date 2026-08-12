"""Phase 4.6: combined candidate after the isolated ablations."""

_base_ = ['./_base_phase4_finetune.py']

crop_size = (518, 518)

custom_imports = dict(
    imports=[
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc',
        'projects.Ultrasound_Foundation_multitask.mmseg.evaluation.organs_metric',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.multitask_encoder_decoder',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.cls_head',
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.transforms',
        'projects.Ultrasound_Foundation_multitask.configs.phase4.repeat_balanced_uusivc_dataset',
    ])

work_dir = (
    '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/'
    'phase4_06_combined_candidate')

aspect_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=0, seg=255)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegClsInputs'),
]

aspect_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='Pad', size=crop_size, pad_val=dict(img=0, seg=255)),
    dict(type='PackSegClsInputs'),
]

train_pipeline = aspect_train_pipeline
test_pipeline = aspect_test_pipeline
train_dataloader = dict(
    dataset=dict(
        type='RepeatBalancedUUSIVCSEGClsDataset',
        pipeline=aspect_train_pipeline,
        task_repeat_factors={
            'image_cls': 3,
            'image_seg': 2,
            'ceus_cls': 1,
            'ceus_seg': 1,
            'video_seg': 1},
        organ_repeat_factors={
            'Liver_CEUS': 2,
            'Thyroid_US': 2,
            'Cardiac_US': 2},
        class_repeat_factors={
            'Breast_CEUS/1': 2,
            'Liver_CEUS/0': 5,
            'Prostate_CEUS/1': 3}))
val_dataloader = dict(dataset=dict(pipeline=aspect_test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=aspect_test_pipeline))

model = dict(
    decode_cls_head=dict(
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    decode_head=dict(
        train_cfg=dict(
            num_points=24576,
            oversample_ratio=4.0,
            importance_sample_ratio=0.9)))

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=32000,
    val_interval=4000)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=32000,
        by_epoch=False),
]
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='Challenge/Overall',
        rule='greater',
        max_keep_ckpts=6))

visualizer = dict(
    type='SegLocalVisualizer',
    name='phase4_06_combined_candidate',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend',
            name='phase4_06_combined_candidate'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='UUSIVC-mmseg-phase4',
                name='phase4_06_combined_candidate',
                tags=['phase4', 'combined']))])
