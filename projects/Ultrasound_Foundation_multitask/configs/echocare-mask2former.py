_base_ = [
    # '../../../configs/_base_/models/pspnet_r50-d8.py',
    # './_base_/models/pspnet_r50_multitask.py',
    './_base_/models/echocare-mask2former.py',
    './_base_/datasets/uusivc_dataset.py',
    '../../../configs/_base_/default_runtime.py',
    # '../../../configs/_base_/schedules/schedule_20k.py'
]
work_dir = '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/echocare-mask2former'

custom_imports = dict(
    imports=['projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc',
             'projects.Ultrasound_Foundation_multitask.mmseg.evaluation.organs_metric',
             'projects.Ultrasound_Foundation_multitask.mmseg.models.multitask_encoder_decoder',
             'projects.Ultrasound_Foundation_multitask.mmseg.models.cls_head',
             'projects.Ultrasound_Foundation_multitask.mmseg.datasets.transforms',
             'projects.Ultrasound_Foundation_multitask.mmseg.backbones.echocare_swin',
             ],)
# crop_size = (512, 1024)
crop_size = (512, 512)
# data_preprocessor = dict(size=crop_size)
# model = dict(data_preprocessor=data_preprocessor, # data preprocessor of pspnet is useing normalization of bdd100k
#              decode_head=dict(num_classes=2), decode_cls_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
# train_dataloader = dict(batch_size=1, num_workers=1)

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=4)# for debugging validation

depths = [2, 2, 18, 2]
# dataset config
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     # dict(
#     #     type='RandomChoiceResize',
#     #     scales=[int(x * 0.1 * 640) for x in range(5, 21)],
#     #     resize_type='ResizeShortestEdge',
#     #     max_size=2560),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]
train_dataloader = dict(batch_size=4, num_workers=1)

data_preprocessor = dict(size=crop_size)
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
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(
#         type='CheckpointHook', by_epoch=False, interval=5000,
#         save_best='mIoU'),
#     sampler_seed=dict(type='DistSamplerSeedHook'),)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
