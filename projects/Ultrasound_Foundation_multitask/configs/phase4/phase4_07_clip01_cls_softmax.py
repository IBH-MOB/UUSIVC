"""Phase 4.7: softmax classification fine-tune with clip norm 0.1."""

_base_ = ['./phase4_01_cls_softmax_finetune.py']

work_dir = (
    '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/'
    'phase4_07_clip01_cls_softmax')

optim_wrapper = dict(clip_grad=dict(max_norm=0.1, norm_type=2))

visualizer = dict(
    type='SegLocalVisualizer',
    name='phase4_07_clip01_cls_softmax',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend',
            name='phase4_07_clip01_cls_softmax'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='UUSIVC-mmseg-phase4',
                name='phase4_07_clip01_cls_softmax',
                tags=['phase4', 'classification', 'softmax', 'clip-0.1']))])
