_base_ = ['./e5_bestcombo.py']

work_dir = '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/swinB-mask2former_e6_pmd_usnorm'
crop_size = (384,384)
# Same winning pipeline as e5_bestcombo (PMD + flip, no rotate/clahe/cutout/
# gamma) but with dataset-derived z-score normalization instead of ImageNet.
# E1 showed dataset norm helps CLS_AUC (0.863 vs 0.837) and rescues Appendix
# AUC, but hurt seg mIoU under rotate+gamma. This run isolates the norm swap
# under the clean PMD pipeline to see if the seg regression was caused by the
# old augs rather than the norm itself.
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.157926 * 255, 0.157926 * 255, 0.157926 * 255],
    std=[0.199434 * 255, 0.199434 * 255, 0.199434 * 255],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(data_preprocessor=data_preprocessor)

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
