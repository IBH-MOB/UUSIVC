# dataset settings
dataset_type = 'UUSIVCSEGClsDataset'
# data_root = '/scratch/dr/m.badran/UUSIC/dataset-challenge/Extracted/TRAIN/Challenge_Data_Public/image_cls'
data_root_train = '/scratch/dr/UUSIVC26/mmseg_format_full/train'
data_root_val = '/scratch/dr/UUSIVC26/mmseg_format_full/val'

# data_root='/scratch/dr/m.badran/UUSIC/dataset-challenge-sample'

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(
    #     type='RandomResize',
    #     scale=(2048, 1024),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegClsInputs'),
        # meta_keys=('organ', 'img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label')),
    # dict(type='PackClsInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegClsInputs'),
        # meta_keys=('organ', 'img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label')),
    # dict(type='PackClsInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegClsInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True, seed =42),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_train,
        # data_prefix=dict(
        #     img_path='images/10k/train',
        #     seg_map_path='labels/sem_seg/masks/train'),
        pipeline=train_pipeline,
        reduce_zero_label=False,
        label_map=dict({255: 1})))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, seed =42),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_val,
        # data_prefix=dict(
        #     img_path='images/10k/val',
        #     seg_map_path='labels/sem_seg/masks/val'),
        pipeline=test_pipeline,
        reduce_zero_label=False,
        label_map=dict({255: 1})))
test_dataloader = val_dataloader

val_evaluator = dict(type='OrgansIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                     data_root=data_root_val)
test_evaluator = val_evaluator
