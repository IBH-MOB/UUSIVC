_base_ = [
    # '../../../configs/_base_/models/pspnet_r50-d8.py',
    # './_base_/models/pspnet_r50_multitask.py',
    './_base_/models/swin-mask2former.py',
    './_base_/datasets/uusivc_dataset.py',
    '../../../configs/_base_/default_runtime.py',
    # '../../../configs/_base_/schedules/schedule_20k.py'
]
work_dir = '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/swinB-mask2former'

custom_imports = dict(
    imports=['projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc',
             'projects.Ultrasound_Foundation_multitask.mmseg.evaluation.organs_metric',
             'projects.Ultrasound_Foundation_multitask.mmseg.models.multitask_encoder_decoder',
             'projects.Ultrasound_Foundation_multitask.mmseg.models.cls_head',
             'projects.Ultrasound_Foundation_multitask.mmseg.datasets.transforms',
             ],)

crop_size = (384, 384)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegClsInputs'),
]

depths = [2, 2, 18, 2]
train_dataloader = dict(batch_size=16, num_workers=10,
                        dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(data_preprocessor=data_preprocessor, # data preprocessor of pspnet is useing normalization of bdd100k
             decode_head=dict(num_classes=2), decode_cls_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend',
            name=work_dir.split('/')[-1],
        ),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='UUSIVC-mmseg',
                name=work_dir.split('/')[-1],
            )
        )
    ],
    name='visualizer'
)
auto_scale_lr = dict(enable=False, base_batch_size=16)
