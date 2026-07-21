
_base_ = [
    # '../../../configs/_base_/models/pspnet_r50-d8.py',
    './_base_/models/pspnet_r50_multitask.py',
    './_base_/datasets/uusivc_dataset.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_160k.py'
]
work_dir = '/scratch/dr/m.badran/UUSIC/mmseg/work_dirs/uusivc_multitask_pspnet_r50_crop256_vitclshead'  

custom_imports = dict(
    imports=['projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc',
             'projects.Ultrasound_Foundation_multitask.mmseg.evaluation.organs_metric',
             'projects.Ultrasound_Foundation_multitask.mmseg.models.multitask_encoder_decoder',
             'projects.Ultrasound_Foundation_multitask.mmseg.models.cls_head',
             'projects.Ultrasound_Foundation_multitask.mmseg.datasets.transforms',
             'mmpretrain.models.heads.vision_transformer_head'
             ],)
# crop_size = (1024, 1024)
# crop_size = (512, 512)
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor, # data preprocessor of pspnet is useing normalization of bdd100k
             decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2),
                 decode_cls_head=dict(
                    _delete_=True,
                    type='VisionTransformerClsHead',
                    in_channels=2048,
                    num_classes=1,
                    # norm_cfg=dict(type='SyncBN', requires_grad=True),
                    loss=dict(
                        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                    _scope_ = "mmpretrain",),
                neck_cls=dict(
                    type='GlobalAveragePooling',
                    _scope_ = "mmpretrain",),
            )

# print("model['decode_cls_head'] before pop:", model['decode_cls_head'])
# model['decode_cls_head'].pop("in_index").pop("channels").pop("num_convs").pop("concat_input").pop("dropout_ratio").pop("norm_cfg")

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
train_dataloader = dict(batch_size=4, num_workers=4)

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=4)# for debugging validation

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
                project='UUSIVC-mmseg-multitask',
                name=work_dir.split('/')[-1],
            )
        )
    ],
    name='visualizer'
)
