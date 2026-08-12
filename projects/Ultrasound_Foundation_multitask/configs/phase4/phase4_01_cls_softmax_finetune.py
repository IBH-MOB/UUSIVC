"""Phase 4.1: replace the mismatched two-logit BCE objective with softmax CE."""

_base_ = ['./_base_phase4_finetune.py']

work_dir = (
    '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/'
    'phase4_01_cls_softmax_finetune')

model = dict(
    decode_cls_head=dict(
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)))

visualizer = dict(
    type='SegLocalVisualizer',
    name='phase4_01_cls_softmax_finetune',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend',
            name='phase4_01_cls_softmax_finetune'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='UUSIVC-mmseg-phase4',
                name='phase4_01_cls_softmax_finetune',
                tags=['phase4', 'classification', 'softmax']))])
