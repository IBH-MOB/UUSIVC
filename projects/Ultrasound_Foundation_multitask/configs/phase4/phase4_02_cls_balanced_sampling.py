"""Phase 4.2: softmax CE plus organ/class repeat balancing."""

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
    'phase4_02_cls_balanced_sampling')

model = dict(
    decode_cls_head=dict(
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)))

train_dataloader = dict(
    dataset=dict(
        type='RepeatBalancedUUSIVCSEGClsDataset',
        task_repeat_factors=dict(image_cls=2),
        class_repeat_factors={
            'Breast_CEUS/1': 2,
            'Liver_CEUS/0': 5,
            'Prostate_CEUS/1': 3,
        }))

visualizer = dict(
    type='SegLocalVisualizer',
    name='phase4_02_cls_balanced_sampling',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend',
            name='phase4_02_cls_balanced_sampling'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='UUSIVC-mmseg-phase4',
                name='phase4_02_cls_balanced_sampling',
                tags=['phase4', 'classification', 'balanced']))])
