"""Repeat-based balancing for Phase 4 ablations.

The original dataset is flat and is sampled uniformly. This wrapper keeps the
same sample format while repeating entries according to task, organ, and class
factors. It is intentionally deterministic at config-build time.
"""

import math
import os

import torch
from mmseg.registry import DATASETS

from projects.Ultrasound_Foundation_multitask.mmseg.datasets.uusivc import (
    UUSIVCSEGClsDataset)


@DATASETS.register_module()
class RepeatBalancedUUSIVCSEGClsDataset(UUSIVCSEGClsDataset):

    def __init__(self,
                 task_repeat_factors=None,
                 organ_repeat_factors=None,
                 class_repeat_factors=None,
                 **kwargs):
        self.task_repeat_factors = dict(task_repeat_factors or {})
        self.organ_repeat_factors = dict(organ_repeat_factors or {})
        self.class_repeat_factors = dict(class_repeat_factors or {})
        super().__init__(**kwargs)

    def _build_data_list(self):
        fallback_mask = '/scratch/dr/m.badran/UUSIC/grayscale_10x10.png'
        base_list = []
        for organ in self.organs:
            folder = os.path.join(self.data_root, organ)
            mask_folder = os.path.join(folder, 'mask')
            mask_names = set()
            if os.path.isdir(mask_folder):
                mask_names = {
                    entry.name for entry in os.scandir(mask_folder)
                    if entry.name.endswith('.png')
                }

            for label in ('0', '1', '2'):
                label_folder = os.path.join(folder, label)
                if not os.path.isdir(label_folder):
                    continue
                for entry in os.scandir(label_folder):
                    if not entry.name.endswith('.png'):
                        continue
                    image_path = entry.path
                    mask_path = (
                        os.path.join(mask_folder, entry.name)
                        if entry.name in mask_names else fallback_mask)
                    base_list.append(
                        dict(
                            img_path=image_path,
                            seg_map_path=mask_path,
                            is_npy=image_path.endswith('.npy'),
                            label_map=self.label_map,
                            reduce_zero_label=self.reduce_zero_label,
                            seg_fields=[],
                            organ=organ,
                            gt_label=torch.tensor([int(label)])))

        repeated = []

        for item in base_list:
            organ = item['organ']
            label = int(item['gt_label'][0].item())
            image_name = os.path.basename(item['img_path'])
            mask_path = os.path.join(
                self.data_root, organ, 'mask', image_name)
            has_mask = os.path.isfile(mask_path)

            if organ.endswith('_Vids'):
                tasks = ['video_seg']
            else:
                tasks = []
                if label != 2:
                    tasks.append(
                        'ceus_cls' if organ.endswith('_CEUS') else 'image_cls')
                if has_mask:
                    tasks.append(
                        'ceus_seg' if organ.endswith('_CEUS') else 'image_seg')

            factors = [
                self.task_repeat_factors.get(task, 1.0)
                for task in tasks
            ]
            factors.append(self.organ_repeat_factors.get(organ, 1.0))
            if label != 2:
                factors.append(
                    self.class_repeat_factors.get(
                        f'{organ}/{label}', 1.0))

            repeat = max(1, int(math.ceil(max(factors))))
            repeated.extend(
                dict(item, seg_fields=list(item['seg_fields']))
                for _ in range(repeat))

        return repeated
