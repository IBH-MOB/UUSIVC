"""Phase 4.4: keep native geometry and pad to the ViT input size."""

_base_ = ['./_base_phase4_finetune.py']

crop_size = (518, 518)

work_dir = (
    '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/'
    'phase4_04_aspect_preserving')

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

train_dataloader = dict(dataset=dict(pipeline=aspect_train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=aspect_test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=aspect_test_pipeline))

visualizer = dict(
    type='SegLocalVisualizer',
    name='phase4_04_aspect_preserving',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend',
            name='phase4_04_aspect_preserving'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='UUSIVC-mmseg-phase4',
                name='phase4_04_aspect_preserving',
                tags=['phase4', 'aspect-preserving']))])
