_base_ = [
    # '../../../configs/_base_/models/pspnet_r50-d8.py',
    './_base_/models/upernet_swin.py',
    './_base_/datasets/uusivc_dataset.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_160k.py'
]
work_dir = '/scratch/dr/m.badran/UUSIC/mmseg/work_dirs/uusivc_multitask_upernet_swin'

custom_imports = dict(
    imports=['projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc',
             'projects.Ultrasound_Foundation_multitask.mmseg.evaluation.organs_metric',
             'projects.Ultrasound_Foundation_multitask.mmseg.models.multitask_encoder_decoder',
             'projects.Ultrasound_Foundation_multitask.mmseg.models.cls_head',
             'projects.Ultrasound_Foundation_multitask.mmseg.datasets.transforms'
             ],)
# crop_size = (512, 1024)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor, # data preprocessor of pspnet is useing normalization of bdd100k
             decode_head=dict(num_classes=2), decode_cls_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))

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
