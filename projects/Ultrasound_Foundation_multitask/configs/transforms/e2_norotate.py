_base_ = ['./_base_swin_exp.py']

crop_size = (384, 384)

work_dir = '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/swinB-mask2former_e2_norotate'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='BioMedicalRandomGamma', prob=0.5, gamma_range=(0.5, 2.0),
        per_channel=False),
    dict(type='PackSegClsInputs'),
]

train_dataloader = dict(
    batch_size=4, num_workers=2, dataset=dict(pipeline=train_pipeline))

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend', name=work_dir.split('/')[-1]),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='UUSIVC-mmseg', name=work_dir.split('/')[-1])),
    ],
    name='visualizer')