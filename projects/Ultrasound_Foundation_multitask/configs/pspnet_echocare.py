_base_ = [
    # '../../../configs/_base_/models/pspnet_r50-d8.py',
    './_base_/models/pspnet-echocare.py',
    './_base_/datasets/uusivc_dataset.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_80k.py'
]
work_dir = '/scratch/dr/o.iraqy/UUSIVC-MMSeg/work_dirs/pspnet_echocare'  

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
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor, # data preprocessor of pspnet is useing normalization of bdd100k
             decode_head=dict(num_classes=2), decode_cls_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))

# train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
train_dataloader = dict(batch_size=4, num_workers=1)

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
                project='UUSIVC-mmseg',
                name=work_dir.split('/')[-1],
            )
        )
    ],
    name='visualizer'
)
