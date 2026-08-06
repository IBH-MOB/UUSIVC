# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import json
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS
import logging

from torch import Tensor
from mmengine.structures import BaseDataElement

from sklearn.metrics import roc_auc_score
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt


# CEUS frames are down-weighted so one video contributes the same as one image
# instance. Cardiac video segmentation intentionally remains frame-weighted.
_FRAME_WEIGHTED_TASKS = ('ceus_cls', 'ceus_seg')

# Dataset supports used when combining per-dataset scores into task scores.
_SECOND_SET_SUPPORT = {
    'image_cls Appendix': 75,
    'image_cls Breast': 176,
    'image_cls Liver': 121,
    'image_cls Prostate': 252,
    'ceus_cls BreastCEUS': 41,
    'ceus_cls LiverCEUS': 44,
    'ceus_cls ProstateCEUS': 45,
    'ceus_cls ThyroidCEUS': 40,
    'image_seg Breast': 176,
    'image_seg Breast_luminal': 256,
    'image_seg Fetal_Head': 142,
    'image_seg Heart': 90,
    'image_seg Kidney': 79,
    'image_seg Prostate': 251,
    'image_seg Thyroid': 527,
    'ceus_seg BreastCEUS': 205,
    'ceus_seg LiverCEUS': 201,
    'ceus_seg ProstateCEUS': 224,
    'ceus_seg ThyroidCEUS': 199,
    'video_seg CardiacCH': 690,
}

# The evaluation folders use training names, while second_set uses challenge
# dataset names. The task is part of the key because one folder can provide
# both classification and segmentation metrics.
_SECOND_SET_DATASET_NAMES = {
    ('image_cls', 'Appendix_US'): 'Appendix',
    ('image_cls', 'Breast_US'): 'Breast',
    ('image_cls', 'Liver_US'): 'Liver',
    ('image_cls', 'Prostate_US'): 'Prostate',
    ('ceus_cls', 'Breast_CEUS'): 'BreastCEUS',
    ('ceus_cls', 'Liver_CEUS'): 'LiverCEUS',
    ('ceus_cls', 'Prostate_CEUS'): 'ProstateCEUS',
    ('ceus_cls', 'Thyroid_CEUS'): 'ThyroidCEUS',
    ('image_seg', 'Breast_US'): 'Breast',
    ('image_seg', 'BreastLuminal_US'): 'Breast_luminal',
    ('image_seg', 'FetalHead_US'): 'Fetal_Head',
    ('image_seg', 'Cardiac_US'): 'Heart',
    ('image_seg', 'Kidney_US'): 'Kidney',
    ('image_seg', 'Prostate_US'): 'Prostate',
    ('image_seg', 'Thyroid_US'): 'Thyroid',
    ('ceus_seg', 'Breast_CEUS'): 'BreastCEUS',
    ('ceus_seg', 'Liver_CEUS'): 'LiverCEUS',
    ('ceus_seg', 'Prostate_CEUS'): 'ProstateCEUS',
    ('ceus_seg', 'Thyroid_CEUS'): 'ThyroidCEUS',
    ('video_seg', 'Cardiac_Vids'): 'CardiacCH',
}


def _second_set_support(task: str, organ_folder: str) -> int:
    """Return the configured second-set support for an evaluation folder."""
    dataset_name = _SECOND_SET_DATASET_NAMES.get((task, organ_folder))
    if dataset_name is None:
        raise KeyError(
            f'No second_set support configured for task={task!r}, '
            f'organ_folder={organ_folder!r}')
    support_key = f'{task} {dataset_name}'
    try:
        return _SECOND_SET_SUPPORT[support_key]
    except KeyError as exc:
        raise KeyError(
            f'No second_set support configured for {support_key!r}') from exc


def _task_of(organ_folder: str, is_seg: bool) -> Optional[str]:
    """Map an organ folder name + cls/seg side to one of the 5 task names."""
    if organ_folder is None:
        return None
    if organ_folder.endswith('_US'):
        return 'image_seg' if is_seg else 'image_cls'
    if organ_folder.endswith('_CEUS'):
        return 'ceus_seg' if is_seg else 'ceus_cls'
    if organ_folder.endswith('_Vids'):
        return 'video_seg' if is_seg else None
    return None


def _resolve_frame_weights(data_root: str) -> Dict[str, float]:
    """Return ``{task: n_videos / n_frames}`` for CEUS tasks.

    Each CEUS frame is weighted by ``1 / frames_per_video`` so the total weight
    of one video equals one instance. Image tasks and ``video_seg`` get weight
    1.0 implicitly because they are not present in this dict.
    """
    n_vid: Dict[str, int] = defaultdict(int)
    n_fr: Dict[str, int] = defaultdict(int)
    if not data_root or not os.path.isdir(data_root):
        return {}
    for folder in sorted(os.listdir(data_root)):
        mp = os.path.join(data_root, folder, 'mappings.json')
        if not os.path.isfile(mp):
            continue
        try:
            mapping = json.load(open(mp))
        except Exception:  # noqa: BLE001
            continue
        seen_vid: Dict[str, set] = defaultdict(set)
        for _fname, e in mapping.items():
            if not e.get('is_video'):
                continue
            t = e.get('task')
            if t not in _FRAME_WEIGHTED_TASKS:
                continue
            vid = e.get('video_id')
            n_fr[t] += 1
            if vid not in seen_vid[t]:
                seen_vid[t].add(vid)
                n_vid[t] += 1
    return {t: (n_vid[t] / n_fr[t] if n_fr[t] else 1.0) for t in n_fr}

@METRICS.register_module()
class OrgansIoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 data_root: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.results_seg_iou = dict()
        self.results_seg_nsd = dict()
        self.results_seg_dice = dict()   # per-case Dice, parallel to seg_iou/seg_nsd
        self.results_seg_w = dict()      # per-frame weight, parallel to seg_iou
        self.results_cls_acc = dict()
        self.results_cls_auc = dict()
        self.results_cls_w = dict()      # per-frame weight, parallel to cls_acc
        # Per-task frame weight = n_videos / n_frames for CEUS tasks only.
        # ``video_seg`` intentionally remains at the normal frame weight of 1.
        self.frame_weights = _resolve_frame_weights(data_root) if data_root else {}
        if self.frame_weights:
            print_log(
                f'{self.__class__.__name__} instance-level frame weights: '
                + ', '.join(f'{t}={w:.4f}' for t, w in sorted(self.frame_weights.items())),
                logger='current', level=logging.INFO)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            pred_cls = data_sample['cls_logits']
            gt_cls = data_sample['gt_label'][0].to(pred_label.device)
            organ = data_sample['organ']
            # Only CEUS tasks use instance-level frame weighting. Image tasks and
            # video_seg retain the normal per-sample weight of 1.0.
            cls_task = _task_of(organ, is_seg=False)
            seg_task = _task_of(organ, is_seg=True)
            w_cls = (self.frame_weights.get(cls_task, 1.0)
                     if cls_task in _FRAME_WEIGHTED_TASKS else 1.0)
            w_seg = (self.frame_weights.get(seg_task, 1.0)
                     if seg_task in _FRAME_WEIGHTED_TASKS else 1.0)
            if organ not in self.results_seg_iou:
                self.results_seg_iou[organ] = []
                self.results_seg_nsd[organ] = []
                self.results_seg_dice[organ] = []
                self.results_seg_w[organ] = []
            if organ not in self.results_cls_acc:
                self.results_cls_acc[organ] = []
                self.results_cls_auc[organ] = []
                self.results_cls_w[organ] = []

            if gt_cls != 2:
                self.results_cls_acc[organ].append(
                    float((pred_cls > 0.5)[gt_cls]))
                self.results_cls_auc[organ].append(
                    [gt_cls.cpu(), pred_cls.cpu()[1].item()])
                self.results_cls_w[organ].append(w_cls)

            if (data_sample['gt_sem_seg']['data']==1).all():
                if pred_label.shape == data_sample['gt_sem_seg']['data'].squeeze().shape:
                    print("ERROR Skipping IMG in VAL that should not be skipped")
                # print("skipping")
                continue ## dont eval for seg
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)

                area_intersect, area_union, area_pred_label, area_label = \
                    self.intersect_and_union(pred_label, label, num_classes,
                                              self.ignore_index)
                self.results_seg_iou[organ].append(
                    (area_intersect, area_union, area_pred_label, area_label))
                # Per-case DSC: computed for THIS sample only (not pooled across
                # the dataset), matching how NSD is computed per-case below.
                self.results_seg_dice[organ].append(
                    self.case_dice(area_intersect, area_pred_label, area_label))
                self.results_seg_nsd[organ].append(
                    compute_nsd(label, pred_label))
                self.results_seg_w[organ].append(w_seg)
                
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            # if key == 'aAcc':
            #     metrics[key] = val
            # else:
            #     metrics['m' + key] = val
            metrics[key] = val

        # each class table
        ret_metrics.pop('SEG_aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    def compute_metrics_weighted(self, results: list,
                                 weights: List[float]) -> Dict[str, float]:
        """Weighted-micro version of compute_metrics.

        Each per-sample (intersect, union, pred_label, label) tuple is scaled by
        its task-specific frame weight before summing.

        NOTE: This produces pooled/micro versions of mIoU / mFscore (area-summed
        across all cases first, then divided). SEG_mDice is intentionally
        overwritten later in ``evaluate()`` with a per-case, support-weighted
        average instead of this pooled value, to match the official scoring
        definition (per-case DSC averaged per dataset, weighted by support).
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            return OrderedDict()
        results = tuple(zip(*results))
        assert len(results) == 4
        wt = torch.as_tensor(weights, dtype=torch.float32)
        # Stack per-sample tensors and scale by weights.
        inter = torch.stack(results[0])          # [N, C]
        union = torch.stack(results[1])
        pred = torch.stack(results[2])
        labl = torch.stack(results[3])
        total_area_intersect = (inter * wt[:, None]).sum(0)
        total_area_union = (union * wt[:, None]).sum(0)
        total_area_pred_label = (pred * wt[:, None]).sum(0)
        total_area_label = (labl * wt[:, None]).sum(0)
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)
        ret_metrics_summary = OrderedDict({
            m: np.round(np.nanmean(v) * 100, 2)
            for m, v in ret_metrics.items()
        })
        return {k: v for k, v in ret_metrics_summary.items()}

    @staticmethod
    def case_dice(area_intersect: torch.Tensor, area_pred_label: torch.Tensor,
                  area_label: torch.Tensor) -> float:
        """Per-case Dice Similarity Coefficient (DSC).

        Computes 2*intersect/(pred+label) per class for THIS SINGLE SAMPLE,
        then averages across the classes that are actually present (in either
        prediction or ground truth) for this case. This mirrors how NSD is
        already computed per-case, so DSC and NSD are combined consistently
        before being averaged/weighted per dataset.

        Args:
            area_intersect (torch.Tensor): Per-class intersection area for one
                sample, shape (num_classes,).
            area_pred_label (torch.Tensor): Per-class predicted area for one
                sample, shape (num_classes,).
            area_label (torch.Tensor): Per-class ground-truth area for one
                sample, shape (num_classes,).

        Returns:
            float: Scalar per-case Dice score (NaN if no class is present).
        """
        denom = area_pred_label + area_label
        valid = denom > 0
        if not valid.any():
            return float('nan')
        dice_per_class = 2 * area_intersect[valid] / denom[valid]
        return float(dice_per_class.mean().item())

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'SEG_aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['SEG_mIoU'] = iou
                ret_metrics['SEG_mAcc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['SEG_mDice'] = dice
                ret_metrics['SEG_mAcc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['SEG_mFscore'] = f_value
                ret_metrics['SEG_mPrecision'] = precision
                ret_metrics['SEG_mRecall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics

    
    
    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Reports per-organ metrics, an ``Overall_US`` micro-average across all
        samples (CEUS frame-weighted; video_seg frame-weighted normally),
        per-task component metrics
        (``image_cls``, ``image_seg``, ``ceus_cls``, ``ceus_seg``,
        ``video_seg``), five ``Average/<task>`` scalars, and the
        ``Challenge/{Classification,Segmentation,Overall}`` composites which are
        macro-averaged across tasks so ceus/video are not drowned out by image
        volume. CEUS frames are down-weighted to one instance per video
        (``n_videos / n_frames`` from the val mapping), while video_seg frames
        retain weight 1.0.

        DSC (``SEG_mDice``) and NSD (``SEG_NSD``) are both computed per-case
        (per image/frame), then combined into each dataset's score as a
        support-weighted average of those per-case values, per the official
        scoring definition.
        """
        metrics = None

        # --- Build the Overall_US bucket (micro across all samples) ---------
        overall_seg_iou, overall_seg_nsd, overall_seg_dice, overall_seg_w = \
            [], [], [], []
        for organ in self.results_seg_iou:
            if self.results_seg_iou[organ]:
                overall_seg_iou.extend(self.results_seg_iou[organ])
                overall_seg_nsd.extend(self.results_seg_nsd[organ])
                overall_seg_dice.extend(self.results_seg_dice[organ])
                overall_seg_w.extend(self.results_seg_w[organ])
        self.results_seg_iou['Overall_US'] = overall_seg_iou
        self.results_seg_nsd['Overall_US'] = overall_seg_nsd
        self.results_seg_dice['Overall_US'] = overall_seg_dice
        self.results_seg_w['Overall_US'] = overall_seg_w
        overall_cls_acc, overall_cls_auc, overall_cls_w = [], [], []
        for organ in self.results_cls_acc:
            if self.results_cls_acc[organ]:
                overall_cls_acc.extend(self.results_cls_acc[organ])
                overall_cls_auc.extend(self.results_cls_auc[organ])
                overall_cls_w.extend(self.results_cls_w[organ])
        self.results_cls_acc['Overall_US'] = overall_cls_acc
        self.results_cls_auc['Overall_US'] = overall_cls_auc
        self.results_cls_w['Overall_US'] = overall_cls_w

        # --- Build the 5 per-task buckets (micro per task) ------------------
        task_seg = defaultdict(lambda: {'iou': [], 'nsd': [], 'dice': [], 'w': []})
        task_cls = defaultdict(lambda: {'acc': [], 'auc': [], 'w': []})
        for organ in self.results_seg_iou:
            if organ == 'Overall_US':
                continue
            t = _task_of(organ, is_seg=True)
            if t and self.results_seg_iou[organ]:
                task_seg[t]['iou'].extend(self.results_seg_iou[organ])
                task_seg[t]['nsd'].extend(self.results_seg_nsd[organ])
                task_seg[t]['dice'].extend(self.results_seg_dice[organ])
                task_seg[t]['w'].extend(self.results_seg_w[organ])
        for organ in self.results_cls_acc:
            if organ == 'Overall_US':
                continue
            t = _task_of(organ, is_seg=False)
            if t and self.results_cls_acc[organ]:
                task_cls[t]['acc'].extend(self.results_cls_acc[organ])
                task_cls[t]['auc'].extend(self.results_cls_auc[organ])
                task_cls[t]['w'].extend(self.results_cls_w[organ])
        for t, v in task_seg.items():
            self.results_seg_iou[t] = v['iou']
            self.results_seg_nsd[t] = v['nsd']
            self.results_seg_dice[t] = v['dice']
            self.results_seg_w[t] = v['w']
            self.results_cls_acc.setdefault(t, [])
            self.results_cls_auc.setdefault(t, [])
            self.results_cls_w.setdefault(t, [])
        for t, v in task_cls.items():
            self.results_cls_acc[t] = v['acc']
            self.results_cls_auc[t] = v['auc']
            self.results_cls_w[t] = v['w']
            self.results_seg_iou.setdefault(t, [])
            self.results_seg_nsd.setdefault(t, [])
            self.results_seg_dice.setdefault(t, [])
            self.results_seg_w.setdefault(t, [])

        # --- Loop over every bucket: organs, Overall_US, and 5 tasks --------
        for organ in list(self.results_seg_iou.keys()):
            if self.collect_device == 'cpu':
                results = collect_results(
                    self.results_seg_iou[organ], size, self.collect_device,
                    tmpdir=self.collect_dir)
                results_seg_nsd = collect_results(
                    self.results_seg_nsd[organ], size, self.collect_device,
                    tmpdir=self.collect_dir)
                results_seg_dice = collect_results(
                    self.results_seg_dice[organ], size, self.collect_device,
                    tmpdir=self.collect_dir)
                results_seg_w = collect_results(
                    self.results_seg_w[organ], size, self.collect_device,
                    tmpdir=self.collect_dir)
                results_cls_acc = collect_results(
                    self.results_cls_acc[organ], size, self.collect_device,
                    tmpdir=self.collect_dir)
                results_cls_auc = collect_results(
                    self.results_cls_auc[organ], size, self.collect_device,
                    tmpdir=self.collect_dir)
                results_cls_w = collect_results(
                    self.results_cls_w[organ], size, self.collect_device,
                    tmpdir=self.collect_dir)
            else:
                results = collect_results(self.results_seg_iou[organ], size, self.collect_device)
                results_seg_nsd = collect_results(self.results_seg_nsd[organ], size, self.collect_device)
                results_seg_dice = collect_results(self.results_seg_dice[organ], size, self.collect_device)
                results_seg_w = collect_results(self.results_seg_w[organ], size, self.collect_device)
                results_cls_acc = collect_results(self.results_cls_acc[organ], size, self.collect_device)
                results_cls_auc = collect_results(self.results_cls_auc[organ], size, self.collect_device)
                results_cls_w = collect_results(self.results_cls_w[organ], size, self.collect_device)

            if is_main_process():
                _metrics = dict()
                seg_n = 0
                if len(results) > 0:
                    results = _to_cpu(results)
                    results_seg_nsd = _to_cpu(results_seg_nsd)
                    results_seg_dice = _to_cpu(results_seg_dice)
                    results_seg_w = _to_cpu(results_seg_w)
                    _metrics = self.compute_metrics_weighted(
                        results, results_seg_w)
                    w = np.asarray(results_seg_w, dtype=np.float64)
                    nsd = np.asarray(results_seg_nsd, dtype=np.float64)
                    dice = np.asarray(results_seg_dice, dtype=np.float64)
                    _metrics['SEG_NSD'] = float(
                        (nsd * w).sum() / w.sum()) if w.sum() > 0 else float('nan')
                    # Per-case DSC, support-weighted per dataset — overrides
                    # the pooled/micro 'SEG_mDice' that compute_metrics_weighted
                    # may have set, to match the official scoring definition
                    # (case/frame score averaged and weighted by support).
                    # NOTE: case_dice() returns a 0-1 value; multiply by 100
                    # here so SEG_mDice stays on the same 0-100 scale as the
                    # rest of the metrics table (SEG_mIoU, SEG_mFscore, ...)
                    # and as seg_comp() in _add_task_and_challenge expects
                    # (it divides SEG_mDice by 100 to normalize against NSD,
                    # which is 0-1). Without this *100, the Dice term gets
                    # divided by 100 twice, crushing every seg-task composite.
                    _metrics['SEG_mDice'] = float(
                        (dice * w).sum() / w.sum() * 100
                    ) if w.sum() > 0 else float('nan')
                    seg_n = float(w.sum())   # instance count (frames, weighted)
                cls_n = 0
                if len(results_cls_acc) > 0:
                    results_cls_acc = _to_cpu(results_cls_acc)
                    results_cls_auc = _to_cpu(results_cls_auc)
                    results_cls_w = _to_cpu(results_cls_w)
                    _metrics.update(self._cls_metrics(
                        results_cls_acc, results_cls_auc, results_cls_w))
                    cls_n = float(
                        np.asarray(results_cls_w, dtype=np.float64).sum())

                if self.prefix:
                    _metrics = {
                        '/'.join((self.prefix, k)): v
                        for k, v in _metrics.items()
                    }
                _metrics = {
                    '/'.join((organ, k)): v
                    for k, v in _metrics.items()
                }
                # Record per-dataset instance counts for presence checks and
                # diagnostics; task aggregation uses second-set supports.
                if seg_n > 0:
                    _metrics[f'{organ}/_seg_n'] = seg_n
                if cls_n > 0:
                    _metrics[f'{organ}/_cls_n'] = cls_n
                if metrics is None:
                    metrics = [_metrics]
                else:
                    metrics[0].update(_metrics)
            else:
                metrics = [None]  # type: ignore

            # reset the results list
            self.results_seg_iou[organ].clear()
            self.results_seg_nsd[organ].clear()
            self.results_seg_dice[organ].clear()
            self.results_seg_w[organ].clear()
            self.results_cls_acc[organ].clear()
            self.results_cls_auc[organ].clear()
            self.results_cls_w[organ].clear()

        if is_main_process():
            metrics = self._add_task_and_challenge(metrics)

        broadcast_object_list(metrics)

        return metrics[0]

    @staticmethod
    def _cls_metrics(accs: List[float], aucs: List[list],
                     weights: List[float]) -> Dict[str, float]:
        """Weighted accuracy + weighted AUC. Returns {} if no samples."""
        out: Dict[str, float] = {}
        if not accs:
            return out
        w = np.asarray(weights, dtype=np.float64)
        a = np.asarray(accs, dtype=np.float64)
        wsum = w.sum()
        out['CLS_Acc'] = float((a * w).sum() / wsum) if wsum > 0 else float('nan')
        if aucs:
            y = np.asarray([r[0] for r in aucs], dtype=np.float64)
            s = np.asarray([r[1] for r in aucs], dtype=np.float64)
            try:
                out['CLS_AUC'] = float(
                    roc_auc_score(y, s, sample_weight=w))
            except ValueError:
                # single-class slice or all-tied scores
                out['CLS_AUC'] = float('nan')
        return out

    def _add_task_and_challenge(self, metrics: list) -> list:
        """Append the 5 Average/<task> scalars and the Challenge/* composites.

        Per the challenge definition:
        - dataset score (cls)  = 0.5 * Accuracy + 0.5 * AUC
        - dataset score (seg)  = 0.7 * DSC + 0.3 * NSD, where DSC and NSD are
          each a support-weighted average of PER-CASE scores (not pooled areas)
        - task score           = second-set-support-weighted average of its
                                datasets
        - overall score        = equal-weighted average of the 5 task scores

        Each organ folder is one dataset. The per-folder component metrics
        (``<folder>/CLS_Acc``, ``<folder>/CLS_AUC``, ``<folder>/SEG_mDice``,
        ``<folder>/SEG_NSD``) and the per-folder weighted instance counts
        (``<folder>/_cls_n``, ``<folder>/_seg_n``) are already in ``metrics[0]``
        from the evaluate loop. Task aggregation uses the hard-coded
        ``_SECOND_SET_SUPPORT`` values instead of those instance counts.
        """
        m = metrics[0]

        def cls_comp(acc, auc):
            return 0.5 * acc + 0.5 * auc

        def seg_comp(dice, nsd):
            # SEG_mDice is on a 0-100 scale (compute_metrics multiplies by 100);
            # SEG_NSD is 0-1.
            return 0.7 * dice / 100.0 + 0.3 * nsd

        # Map each organ folder to its task + per-folder composite + configured
        # second-set support.
        task_folders = defaultdict(list)   # task -> [(score, support), ...]
        for k in list(m.keys()):
            if not k.endswith('/_seg_n') and not k.endswith('/_cls_n'):
                continue
            folder = k.rsplit('/', 1)[0]
            if folder == 'Overall_US':
                # This aggregate bucket ends in ``_US`` but is not a dataset.
                m.pop(k)
                continue
            is_seg = k.endswith('/_seg_n')
            t = _task_of(folder, is_seg=is_seg)
            if t is None:
                continue
            actual_n = m.pop(k)              # weighted count for presence check
            if actual_n <= 0:
                continue
            support = _second_set_support(t, folder)
            if is_seg:
                d_k, ns_k = f'{folder}/SEG_mDice', f'{folder}/SEG_NSD'
                if d_k in m and ns_k in m and not np.isnan(m[d_k]) \
                        and not np.isnan(m[ns_k]):
                    task_folders[t].append(
                        (seg_comp(m[d_k], m[ns_k]), support))
            else:
                a_k, au_k = f'{folder}/CLS_Acc', f'{folder}/CLS_AUC'
                if a_k in m and not np.isnan(m[a_k]):
                    acc = m[a_k]
                    auc = m.get(au_k, float('nan'))
                    # AUC is undefined for single-class slices; fall back to
                    # Accuracy so the dataset still contributes its weight.
                    if auc is None or np.isnan(auc):
                        print_log(
                            f'{self.__class__.__name__}: {folder} AUC '
                            f'undefined (single-class val slice) — using '
                            f'Acc={acc:.4f} as AUC fallback for task {t}.',
                            logger='current', level=logging.WARNING)
                        auc = acc
                    task_folders[t].append((cls_comp(acc, auc), support))

        task_scores: Dict[str, float] = {}
        for t, pairs in task_folders.items():
            total_support = sum(support for _, support in pairs)
            if total_support <= 0:
                continue
            task_scores[t] = sum(s * support for s, support in pairs) \
                / total_support
            m[f'Average/{t}'] = task_scores[t]

        # Diagnostics: cls/seg sub-means (NOT the overall — kept for logging).
        cls_t = [v for k, v in task_scores.items() if k.endswith('_cls')]
        seg_t = [v for k, v in task_scores.items() if k.endswith('_seg')]
        m['Challenge/Classification'] = (
            sum(cls_t) / len(cls_t)) if cls_t else float('nan')
        m['Challenge/Segmentation'] = (
            sum(seg_t) / len(seg_t)) if seg_t else float('nan')
        # Overall = equal-weighted mean of the (up to) 5 task scores.
        all_t = list(task_scores.values())
        m['Challenge/Overall'] = (
            sum(all_t) / len(all_t)) if all_t else float('nan')
        return metrics


def compute_nsd(y_true, y_pred, tolerance=1):
    y_true = (y_true > 0).cpu().numpy().astype(np.uint8)
    y_pred = (y_pred > 0).cpu().numpy().astype(np.uint8)

    boundary_true = find_boundaries(y_true, mode='inner')
    boundary_pred = find_boundaries(y_pred, mode='inner')

    distance_true = distance_transform_edt(1 - boundary_true)
    distance_pred = distance_transform_edt(1 - boundary_pred)

    true_in_pred = (boundary_true & (distance_pred <= tolerance)).sum()
    pred_in_true = (boundary_pred & (distance_true <= tolerance)).sum()

    nsd = (true_in_pred + pred_in_true) / (boundary_true.sum() + boundary_pred.sum() + 1e-6)
    return nsd

def _to_cpu(data: Any) -> Any:
    """Transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data
