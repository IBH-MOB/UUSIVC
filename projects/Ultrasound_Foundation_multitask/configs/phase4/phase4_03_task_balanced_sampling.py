"""Phase 4.3: approximate equal exposure to the five challenge tasks."""

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
    'phase4_03_task_balanced_sampling')

train_dataloader = dict(
    dataset=dict(
        type='RepeatBalancedUUSIVCSEGClsDataset',
        task_repeat_factors={
            'image_cls': 4,
            'image_seg': 3,
            'ceus_cls': 1,
            'ceus_seg': 1,
            'video_seg': 1}))

visualizer = dict(
    type='SegLocalVisualizer',
    name='phase4_03_task_balanced_sampling',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend',
            name='phase4_03_task_balanced_sampling'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='UUSIVC-mmseg-phase4',
                name='phase4_03_task_balanced_sampling',
                tags=['phase4', 'task-balance']))])
