_base_ = ['./vit_small_dinov3.py']

custom_imports = dict(
    imports=[
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc',
        'projects.Ultrasound_Foundation_multitask.mmseg.evaluation.organs_metric',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.multitask_encoder_decoder',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.cls_head',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.organ_cond_cls_head',
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.transforms',
    ])

work_dir = './work_dirs/dinov3_vit_small_mask2former_organ_cond'

model = dict(
    decode_cls_head=dict(
        type='OrganCondCLSHead',
        in_channels=384,
        in_index=3,
        channels=512,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True,
                         loss_weight=1.0),
        organ_names=[
            'Appendix_US', 'Breast_CEUS', 'BreastLuminal_US', 'Breast_US',
            'Cardiac_US', 'Cardiac_Vids', 'FetalHead_US', 'Kidney_US',
            'Liver_CEUS', 'Liver_US', 'Prostate_CEUS', 'Prostate_US',
            'Thyroid_CEUS', 'Thyroid_US',
        ],
        organ_embed_dim=16))

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend',
             name=work_dir.split('/')[-1]),
        dict(type='WandbVisBackend',
             init_kwargs=dict(project='UUSIVC-mmseg',
                              name=work_dir.split('/')[-1])),
    ],
    name='visualizer')
