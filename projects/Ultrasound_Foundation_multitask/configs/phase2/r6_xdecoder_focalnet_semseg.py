"""R6 — XDecoder (FocalNet + XDecoderUnifiedhead) zero-shot semseg on UUSIVC.

XDecoder is a generalized open-vocabulary decoding model (FocalNet backbone
+ XTransformerDecoder + CLIP language encoder). This config runs it in
zero-shot semantic-segmentation mode on the UUSIVC val split using the
text prompt "organ" (XDecoder auto-appends "background").

IMPORTANT — INFERENCE ONLY:
  The mmdetection XDecoder project (`projects/XDecoder/xdecoder/`) ships
  without a `loss` method on `XDecoderUnifiedhead` — only `predict` and
  `post_process` are implemented. There are no training configs in the
  upstream project; all README commands use `dist_test`. This config is
  therefore a **zero-shot evaluation** of whether XDecoder's open-vocab
  pretraining transfers to ultrasound organ segmentation, not a
  finetuning recipe. To run it:

      cd /scratch/dr/o.iraqy/mmdetection
      python tools/test.py \
          /scratch/dr/o.iraqy/UUSIVC-MMSeg/projects/Ultrasound_Foundation_multitask/configs/phase2/r6_xdecoder_focalnet_semseg.py \
          /scratch/dr/o.iraqy/UUSIVC-MMSeg/weights/xdecoder_focalt_last_novg.pt

  (mmseg's tools/train.py cannot drive this — XDecoder lives in mmdet's
  registry and runs under mmdet's default_scope.)

PREREQUISITES:
  - `transformers` (CLIP tokenizer): pip install transformers
    XDecoder's LanguageEncoder imports CLIPTokenizer at module load.
  - Pretrained weights: download first if the GPU node is offline:
      wget https://download.openmmlab.com/mmdetection/v3.0/xdecoder/xdecoder_focalt_last_novg.pt \
           -O /scratch/dr/o.iraqy/UUSIVC-MMSeg/weights/xdecoder_focalt_last_novg.pt
    (xdecoder_focalt_best_openseg.pt is the alternative — better for ADE20K-style semseg).

LIMITATIONS vs the rest of phase2:
  - No training (no loss in upstream XDecoder). To finetune, a `loss`
    method (Hungarian matching + CE + BCE + Dice, Mask2Former-style)
    must be added to XDecoderUnifiedhead — significant code.
  - No cls task. XDecoder semseg is seg-only.
  - No Challenge/Overall composite. Evaluation uses mmdet's SemSegMetric
    (mIoU/mDice/mFscore) — our OrgansIoUMetric is mmseg-scoped and does
    not run under mmdet. Per-organ mIoU is not available without porting
    the metric.
  - Binary seg only (text=["organ"]). Per-organ text prompts
    (e.g. "prostate", "breast") are a possible extension but our seg
    labels are binary, so multi-class argmax would not match.

If zero-shot mIoU is promising, the next step is to implement the loss
and finetune. If not, FocalNet can still be used as a backbone in our
existing Mask2Former + MultitaskEncoderDecoder pipeline (trainable now,
no XDecoder head) — a separate config.
"""
_base_ = [
    '/scratch/dr/o.iraqy/mmdetection/configs/_base_/default_runtime.py',
]

default_scope = 'mmdet'

custom_imports = dict(
    imports=[
        'projects.XDecoder.xdecoder',
        'projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc_mmdet',
    ],
    allow_failed_imports=False)

# ---- Pretrained XDecoder checkpoint (zero-shot) ----
# Download first if offline (see docstring).
load_from = '/scratch/dr/o.iraqy/UUSIVC-MMSeg/weights/xdecoder_focalt_last_novg.pt'

# ---- E6 preprocessing (dataset z-score norm, matches phase2 R1-R5) ----
# DetDataPreprocessor (mmdet) instead of SegDataPreProcessor (mmseg).
# pad_size_divisor=32 because FocalNet + XTransformerEncoder need it.
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0.157926 * 255, 0.157926 * 255, 0.157926 * 255],
    std=[0.199434 * 255, 0.199434 * 255, 0.199434 * 255],
    bgr_to_rgb=True,
    pad_size_divisor=32)

model = dict(
    type='XDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='FocalNet'),
    head=dict(
        type='XDecoderUnifiedhead',
        in_channels=(96, 192, 384, 768),
        pixel_decoder=dict(type='XTransformerEncoderPixelDecoder'),
        transformer_decoder=dict(type='XDecoderTransformerDecoder'),
        task='semseg'),
    test_cfg=dict(mask_thr=0.5, use_thr_for_mc=True, ignore_index=255))

# ---- Dataset: UUSIVC val split via mmdet-compatible adapter ----
crop_size = (384, 384)
data_root_val = '/scratch/dr/UUSIVC26/mmseg_format/val'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_mask=False,
        with_seg=True,
        reduce_zero_label=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'text',
                   'pad_shape', 'padding_size', 'scale_factor', 'flip',
                   'flip_direction'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='UUSIVCSegMMDetDataset',
        data_root=data_root_val,
        data_prefix=dict(img_path=data_root_val, seg_map_path=''),
        return_classes=True,
        seg_label_map={255: 1},
        pipeline=test_pipeline,
        test_mode=True))
val_dataloader = test_dataloader

test_evaluator = dict(type='SemSegMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
val_evaluator = test_evaluator

# Visualization (optional)
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
