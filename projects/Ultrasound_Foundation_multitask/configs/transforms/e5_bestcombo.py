_base_ = ['./_base_swin_exp.py']

crop_size = (384, 384)

work_dir = '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/swinB-mask2former_e5_bestcombo'

# Winners-only pipeline based on experiment results:
#   - B1 (cityscapes/PhotoMetricDistortion) won Overall 0.7733
#   - E2 (no rotation) won Overall 0.7692 and stabilized Breast mIoU
# CLAHE (E3), RandomCutOut (E4), RandomRotate (B0), and BioMedicalRandomGamma
# all regressed vs B1 and are removed here. ImageNet norm is inherited from
# _base_swin_exp.py.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
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
