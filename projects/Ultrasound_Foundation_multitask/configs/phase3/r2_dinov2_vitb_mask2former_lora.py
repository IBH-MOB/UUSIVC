_base_ = ['../phase2/r2_dinov2_vitb_mask2former.py']

custom_imports = dict(
    imports=[
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc',
        'projects.Ultrasound_Foundation_multitask.mmseg.evaluation.organs_metric',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.multitask_encoder_decoder',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.lora_multitask_encoder_decoder',
        'projects.Ultrasound_Foundation_multitask.mmseg.models.cls_head',
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.transforms',
    ])

work_dir = './work_dirs/phase3_r2_dinov2_vitb_lora'

model = dict(
    type='LoRAMultitaskEncoderDecoder',
    lora_cfg=dict(
        rank=64,
        alpha=64,
        drop_rate=0.0,
        block_indices=(6, 7, 8, 9, 10, 11),   # only the 6 later transformer blocks
        targets=[dict(type='qkv')]),    # qkv projections only
    freeze_backbone=True)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'qkv.lora_': dict(lr_mult=1.0),
        },
        norm_decay_mult=0.0))
