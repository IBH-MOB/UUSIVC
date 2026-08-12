"""Phase 4.5: repeat hard organs and increase uncertain mask sampling."""

_base_ = ['./_base_phase4_finetune.py']

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
    'phase4_05_hard_organ_segmentation')

train_dataloader = dict(
    dataset=dict(
        type='RepeatBalancedUUSIVCSEGClsDataset',
        organ_repeat_factors={
            'Liver_CEUS': 2,
            'Thyroid_US': 2,
            'Cardiac_US': 2}))

model = dict(
    decode_head=dict(
        train_cfg=dict(
            num_points=24576,
            oversample_ratio=4.0,
            importance_sample_ratio=0.9)))

visualizer = dict(
    type='SegLocalVisualizer',
    name='phase4_05_hard_organ_segmentation',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend',
            name='phase4_05_hard_organ_segmentation'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='UUSIVC-mmseg-phase4',
                name='phase4_05_hard_organ_segmentation',
                tags=['phase4', 'hard-organ', 'segmentation']))])
