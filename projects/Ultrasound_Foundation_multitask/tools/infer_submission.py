#!/usr/bin/env python
"""Infer a trained UUSIVC multitask model on the OFFICIAL validation set and
produce a challenge-format submission archive.

Reads the official blind val layout produced by the challenge organizer:

    VAL/
      Challenge_Data_Private_v2_fully_anonymized/Val/<task>/<dataset>/<inputs>
      dataset_json_fingerprints_v4/private_val_for_participants.json

and writes a zip with:

    classification.json
    image_seg/<Dataset>/masks/*.png
    ceus_seg/<Dataset>/annotations/*.npz
    video_seg/<Dataset>/annotations/*.npz

Notes:
- ``SegDataPreProcessor`` does NOT resize images; it only normalizes and stacks
  a batch. Resizing to ``crop_size`` is done here (matching the test pipeline),
  and ``ori_shape`` is set to the original frame HxW so the model's
  ``postprocess_result`` resizes the predicted mask back to it.
- CEUS-cls / CEUS-seg are per-video: per-frame predictions are aggregated
  (mean prob for cls, mean prob thresholded for seg) into a single output per
  input ``.npy`` file (one ``classification.json`` entry / one ``mask`` npz).
- video-seg is per-frame: a prediction is written for every index listed in
  the datalist ``frame_indices``, stored as ``fnum_mask`` dict in the npz.
- CardiacCH ``.npy`` files are stored as ``(C, T, H, W)`` float32; CEUS files
  are stored as ``(T, H, W, C)`` uint8.

Example:
    python tools/infer_submission.py \
        --config work_dirs/dinov3_vit_small_mask2former/vit_small_dinov3.py \
        --checkpoint work_dirs/dinov3_vit_small_mask2former/best_Challenge_Overall_iter_160000.pth \
        --data-root /scratch/dr/UUSIVC26/Dataset/Extracted/VAL/Challenge_Data_Private_v2_fully_anonymized/Val \
        --datalist /scratch/dr/UUSIVC26/Dataset/Extracted/VAL/dataset_json_fingerprints_v4/private_val_for_participants.json \
        --out submission.zip
"""
import argparse
import copy
import json
import os
import re
import shutil
import zipfile

import numpy as np
import torch
from PIL import Image

import mmseg  # noqa: F401  registers SegDataPreProcessor etc.
from mmengine.config import Config
from mmengine.registry import MODELS, init_default_scope
from mmseg.structures import SegDataSample
import time

# Map val (task, dataset_name) -> training organ folder (= 'organ' metainfo).
ORGAN_FOLDER = {
    ('image_cls', 'Appendix'):      'Appendix_US',
    ('image_cls', 'Breast'):        'Breast_US',
    ('image_cls', 'Liver'):         'Liver_US',
    ('image_cls', 'Prostate'):      'Prostate_US',
    ('ceus_cls',  'Breast'):        'Breast_CEUS',
    ('ceus_cls',  'Liver'):         'Liver_CEUS',
    ('ceus_cls',  'Prostate'):      'Prostate_CEUS',
    ('ceus_cls',  'Thyroid'):       'Thyroid_CEUS',
    ('image_seg', 'Breast'):        'Breast_US',
    ('image_seg', 'Breast_luminal'):'BreastLuminal_US',
    ('image_seg', 'Cardiac'):       'Cardiac_US',
    ('image_seg', 'Fetal_Head'):    'FetalHead_US',
    ('image_seg', 'Kidney'):        'Kidney_US',
    ('image_seg', 'Prostate'):      'Prostate_US',
    ('image_seg', 'Thyroid'):       'Thyroid_US',
    ('ceus_seg',  'BreastCEUS'):    'Breast_CEUS',
    ('ceus_seg',  'LiverCEUS'):     'Liver_CEUS',
    ('ceus_seg',  'ProstateCEUS'):  'Prostate_CEUS',
    ('ceus_seg',  'ThyroidCEUS'):   'Thyroid_CEUS',
    ('video_seg', 'CardiacCH'):     'Cardiac_Vids',
}


def patch_syncbn(model_cfg):
    """Replace SyncBN with BN for single-process inference (eval-equivalent)."""
    for sub in (model_cfg, model_cfg.get('neck', {}),
                model_cfg.get('decode_head', {}),
                model_cfg.get('decode_cls_head', {}),
                model_cfg.get('auxiliary_head', {})):
        if isinstance(sub, dict) and sub.get('norm_cfg', {}).get('type') == 'SyncBN':
            sub['norm_cfg']['type'] = 'BN'
    pd = model_cfg.get('decode_head', {}).get('pixel_decoder', {})
    if isinstance(pd, dict) and pd.get('norm_cfg', {}).get('type') == 'SyncBN':
        pd['norm_cfg']['type'] = 'BN'
    return model_cfg


def resize_to_crop(frame_np, crop_size):
    """Convert an HWC uint8/float frame to a CHW uint8 tensor resized to crop."""
    if frame_np.ndim == 2:
        img = Image.fromarray(frame_np).convert('RGB')
    elif frame_np.shape[2] == 1:
        img = Image.fromarray(frame_np[..., 0]).convert('RGB')
    else:
        img = Image.fromarray(frame_np.astype(np.uint8)).convert('RGB')
    img = img.resize(crop_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def make_sample(frame_np, organ_folder, crop_size):
    """Build a (tensor, SegDataSample) for a single frame."""
    H, W = frame_np.shape[:2]
    tensor = resize_to_crop(frame_np, crop_size)
    s = SegDataSample()
    s.set_metainfo({
        'organ': organ_folder,
        'ori_shape': (H, W),                       # postprocess resizes back to this
        'img_shape': crop_size,
        'pad_shape': crop_size,
        'scale_factor': (crop_size[1] / W, crop_size[0] / H),
        'flip': False,
    })
    return tensor, s

@torch.no_grad()
def run_predict(model, samples, batch_size):
    """Predict a list of (tensor, sample) sequentially in mini-batches.

    All tensors must already be at ``crop_size`` since ``SegDataPreProcessor``
    does not resize but asserts equal shape inside a batch. Returns the list
    of post-processed SegDataSamples (with pred_sem_seg / cls_logits /
    cls_label) in the original order. Device handling is left to
    ``data_preprocessor.cast_data`` (model + its buffers are on ``device``).
    """
    results = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        batch = {
            'inputs': [t for t, _ in chunk],         # list of CHW tensors
            'data_samples': [s for _, s in chunk],    # list of SegDataSample
        }
        data = model.data_preprocessor(batch)
        preds = model.predict(data['inputs'], data['data_samples'])
        results.extend(preds)
    return results


def load_frames(npy_path, layout_hint, frame_indices=None):
    """Load a video ``.npy`` and return a list of HWC uint8 frames.

    ``layout_hint`` is 'ceus' (= T H W C uint8) or 'video' (= C T H W float32).
    If ``frame_indices`` is None, returns all frames; else those indices.
    """
    arr = np.load(npy_path, allow_pickle=False)
    if layout_hint == 'video':
        # (C, T, H, W) float32 -> (T, H, W, C)
        arr = np.transpose(arr, (1, 2, 3, 0))
    elif arr.ndim == 3:
        # (T, H, W) grayscale -> (T, H, W, 1)
        arr = arr[..., None]
    if frame_indices is not None:
        arr = arr[frame_indices, ...]
    return [arr[i] for i in range(arr.shape[0])]


def fmt_name_from_input(input_name, seg_prefix):
    """Convert an input ``seg_img_00000.png`` / ``seg_video_00000.npy`` /
    ``cls_00000.jpg`` to the submission mask/annotation name stem."""
    m = re.search(r'(\d+)', input_name)
    if m:
        return f'{seg_prefix}_{m.group(1)}'
    return os.path.splitext(input_name)[0]


def main():
    start_time = time.time()
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', default='/scratch/dr/UUSIVC26/Dataset/Extracted/VAL/Challenge_Data_Private_v2_fully_anonymized/Val',
                   help='Root of the official val (the "Val" folder).')
    p.add_argument('--datalist', default='/scratch/dr/UUSIVC26/Dataset/Extracted/VAL/dataset_json_fingerprints_v4/private_val_for_participants.json',
                   help='private_val_for_participants.json path.')
    p.add_argument('--out', default='submission.zip')
    p.add_argument('--work-dir', default='/tmp/uusivc_submission')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Per-(G)PU batch size in frames.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-samples', type=int, default=None)
    args = p.parse_args()

    init_default_scope('mmseg')
    cfg = Config.fromfile(args.config)

    model_cfg = patch_syncbn(copy.deepcopy(cfg.model))
    model = MODELS.build(model_cfg)
    ck = torch.load(args.checkpoint, map_location='cpu')
    sd = ck['state_dict'] if 'state_dict' in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'[warn] missing keys: {len(missing)} (e.g. {missing[:3]})')
    if unexpected:
        print(f'[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})')
    model.eval().to(args.device)

    crop_size = tuple(cfg.data_preprocessor.get('size')
                      or cfg.val_dataloader.dataset.get('pipeline',
                      [{}])[-1].get('scale', (256, 256)))
    print(f'[info] crop_size: {crop_size}')

    with open(args.datalist) as f:
        datalist = json.load(f)
    print(f'[info] datalist entries: {len(datalist)}')

    # group entries by file (so each .npy is iterated only once)
    by_file = {}
    for e in datalist:
        by_file.setdefault(e['input_path_relative'], []).append(e)

    if os.path.exists(args.work_dir):
        shutil.rmtree(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)

    cls_json = {}
    counts = {'image_cls': 0, 'ceus_cls': 0, 'image_seg': 0,
              'ceus_seg': 0, 'video_seg': 0, 'files': 0}
    n_entries = 0

    for input_rel, entries in by_file.items():
        n_entries += 1
        if args.max_samples and n_entries > args.max_samples:
            break
        task = entries[0]['task']
        ds = entries[0]['dataset_name']
        organ = ORGAN_FOLDER.get((task, ds), 'Appendix_US')
        abs_path = os.path.join(args.data_root, input_rel)

        samples = []
        frames_for_out = []  # which frame_indices map to which predicted sample

        if task in ('image_cls', 'image_seg'):
            from PIL import Image as PILImage
            im = PILImage.open(abs_path)
            arr = np.asarray(im.convert('RGB'))
            samples.append(make_sample(arr, organ, crop_size))
            frames_for_out = [None]
        elif task == 'ceus_cls':
            arr = np.load(abs_path, allow_pickle=False)  # (T, H, W, C) uint8
            # use a uniformly-spaced subset of frames to bound the cost.
            T = arr.shape[0]
            idxs = list(range(0, T, max(1, T // 8)))[:8] or [T // 2]
            for ti in idxs:
                samples.append(make_sample(arr[ti], organ, crop_size))
            frames_for_out = idxs
        elif task == 'ceus_seg':
            arr = np.load(abs_path, allow_pickle=False)  # (T, H, W, C) uint8
            for ti in range(arr.shape[0]):
                samples.append(make_sample(arr[ti], organ, crop_size))
            frames_for_out = list(range(arr.shape[0]))
        elif task == 'video_seg':
            fi = entries[0].get('frame_indices') or []
            arr = np.load(abs_path, allow_pickle=False)  # (C, T, H, W) float32
            arr = np.transpose(arr, (1, 2, 3, 0))  # -> T, H, W, C
            for ti in fi:
                samples.append(make_sample(arr[ti], organ, crop_size))
            frames_for_out = list(fi)
        else:
            print(f'[warn] unknown task {task}, skipping {input_rel}')
            continue

        preds = run_predict(model, samples, args.batch_size)

        # ----- write outputs -----
        if task in ('image_cls', 'ceus_cls'):
            # mean probability across frames
            probs = torch.stack([p.cls_logits.float().detach().cpu()
                                 for p in preds], 0).mean(0).numpy()
            probs = probs / max(probs.sum(), 1e-8)
            cls_json[input_rel] = {
                'prediction': int(np.argmax(probs)),
                'probability': [float(probs[0]), float(probs[1])],
            }
            counts[task] += 1
        elif task == 'image_seg':
            p0 = preds[0]
            mask = p0.pred_sem_seg.data.detach().cpu().numpy()
            if mask.ndim == 3: mask = mask[0]
            mask = (mask > 0).astype(np.uint8) * 255
            stem = fmt_name_from_input(entries[0]['input_name'], 'seg_mask')
            out_path = os.path.join(args.work_dir, 'image_seg', ds, 'masks',
                                    f'{stem}.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(mask, mode='L').save(out_path)
            counts[task] += 1
        elif task == 'ceus_seg':
            # aggregate per-frame masks to a single shared 2D mask
            masks = [p.pred_sem_seg.data.detach().cpu().numpy()
                     for p in preds]
            if masks[0].ndim == 3: masks = [m[0] for m in masks]
            stacked = np.stack([(m > 0).astype(np.float32) for m in masks], 0)
            mask = (stacked.mean(0) > 0.5).astype(np.uint8) * 255
            stem = fmt_name_from_input(entries[0]['input_name'], 'seg_annotation')
            out_dir = os.path.join(args.work_dir, 'ceus_seg', ds, 'annotations')
            out_path = os.path.join(out_dir, f'{stem}.npz')
            os.makedirs(out_dir, exist_ok=True)
            np.savez(out_path, mask=mask)
            counts[task] += 1
        elif task == 'video_seg':
            fnum_mask = {}
            for fi, p in zip(frames_for_out, preds):
                mask = p.pred_sem_seg.data.detach().cpu().numpy()
                if mask.ndim == 3: mask = mask[0]
                mask = (mask > 0).astype(np.uint8) * 255
                fnum_mask[str(int(fi))] = mask
            stem = fmt_name_from_input(entries[0]['input_name'], 'seg_annotation')
            out_dir = os.path.join(args.work_dir, 'video_seg', ds, 'annotations')
            out_path = os.path.join(out_dir, f'{stem}.npz')
            os.makedirs(out_dir, exist_ok=True)
            np.savez(out_path, fnum_mask=fnum_mask)
            counts[task] += 1
        counts['files'] += 1
        if counts['files'] % 50 == 0:
            print(f'   ... processed {counts["files"]} / {len(by_file)} files; '
                  f'counts={counts}')

    print(f'[info] done. counts={counts}')

    # write classification.json at top level
    with open(os.path.join(args.work_dir, 'classification.json'), 'w') as f:
        json.dump(cls_json, f, indent=2)
    print(f'[info] classification.json entries: {len(cls_json)}')

    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(args.work_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                zf.write(fp, os.path.relpath(fp, args.work_dir))
    print(f'[done] wrote {args.out}')

    with zipfile.ZipFile(args.out) as zf:
        names = zf.namelist()
    tops = sorted(set(n.split('/')[0] for n in names))
    print('[info] zip top-level:', tops)
    assert set(tops) <= {'classification.json', 'image_seg', 'ceus_seg', 'video_seg'}, tops

    end_time = time.time()
    print(f'[info] total time: {end_time - start_time:.2f} seconds')
    peak_bytes = torch.cuda.max_memory_allocated()
    print(f"Max GPU memory used: {peak_bytes / (1024**2):.2f} MB")


if __name__ == '__main__':
    main()