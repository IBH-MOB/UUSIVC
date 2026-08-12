"""Current-layout evaluation of the historical 112k checkpoint."""

_base_ = [
    '../../../../work_dirs/phase2_r2_dinov2_vitb_mask2former_batch16/'
    'r2_dinov2_vitb_mask2former.py'
]

data_root_val = '/scratch/dr/UUSIVC26/mmseg_format_full/val'
load_from = (
    '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/'
    'phase2_r2_dinov2_vitb_mask2former_batch16/'
    'best_Challenge_Overall_iter_112000.pth')
resume = False
work_dir = (
    '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/'
    'phase4_00_reval_112k')

test_dataloader = dict(dataset=dict(data_root=data_root_val))
test_evaluator = dict(data_root=data_root_val)

visualizer = dict(
    type='SegLocalVisualizer',
    name='phase4_00_reval_112k',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend',
            name='phase4_00_reval_112k'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='UUSIVC-mmseg-phase4',
                name='phase4_00_reval_112k',
                tags=['phase4', 'reval', 'checkpoint-112k']))])
