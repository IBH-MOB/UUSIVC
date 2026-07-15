# Copyright (c) OpenMMLab. All rights reserved.

from ast import Not
import os

from mmseg.datasets.basesegdataset import BaseSegDataset
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from mmengine.dataset.base_dataset import Compose

# from mmseg.registry import DATASETS
# @DATASETS.register_module()

from mmseg.registry import DATASETS

@DATASETS.register_module()
class UUSIVCClsDataset(Dataset):
# class UUSIVCSegDataset(BaseSegDataset):
    metainfo =  dict(
            classes=('Negative', 'Positive'),
            palette=[[128, 200, 128], [0, 0, 70]])
    def __init__(self,
                 data_root: str,
                 **kwargs) -> None:
        self.pipeline = kwargs.get('pipeline', None)
        if not self.pipeline is None:
            self.pipeline = Compose(self.pipeline)

        self.data_root = data_root
        self.organs = [
            name for name in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, name))
        ]
        self.reduce_zero_label = kwargs.get('reduce_zero_label', False)
        self.label_map = kwargs.get('label_map', None)
    
        self.data_list = self._build_data_list()

        # self.dataset_meta = metainfo

    def __len__(self) -> int:
        # return len(self.organs)
        return len(self.data_list)

    

    def _build_data_list(self) -> List[dict]:
        data_root = self.data_root
        data_list: List[dict] = []
        for organ in self.organs:
            # if not "cls" in organ:
            #     continue
            if organ == "Prostate":
                continue

            folder = os.path.join(data_root, organ)
            for i in ["0","1","2"]:
                f = os.path.join(folder, i)
                for img in os.listdir(f):
                    if not img.endswith('.png'):
                        continue
                    img_path = os.path.join(f, img)
                    # seg_map_path = img_path
                    seg_map_path = os.path.join(folder, 'mask', img)
                    if not os.path.exists(seg_map_path):
                        seg_map_path = "/scratch/dr/m.badran/UUSIC/dataset-challenge/Extracted/TRAIN/Challenge_Data_Private_v2_fully_anonymized/Train/image_seg/Breast_luminal/masks/HER2_0_0.png"

                    data_info = dict(
                        img_path=img_path,
                        seg_map_path=seg_map_path,
                        is_npy=img_path.endswith('.npy'),
                        label_map=self.label_map,
                        reduce_zero_label=self.reduce_zero_label,
                        seg_fields=[],
                        # test_mode=test_mode,
                        organ=organ,
                        class_label=int(i)
                    )
                    data_list.append(data_info)
        return data_list



    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.
        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        # data_info = self.get_data_info(idx)
        data_info = idx
        data_info = self.data_list[idx]
        # print(data_info)
        # return data_info
        return self.pipeline(data_info)

        
    def __getitem__(self, idx: int) -> dict:
        return self.prepare_data(idx)


@DATASETS.register_module()
class UUSIVCSegDataset(Dataset):
# class UUSIVCSegDataset(BaseSegDataset):
    metainfo =  dict(
            classes=('background', 'nodule'),
            palette=[[128, 200, 128], [0, 0, 70]])
    def __init__(self,
                 data_root: str,
                 **kwargs) -> None:
        self.pipeline = kwargs.get('pipeline', None)
        if not self.pipeline is None:
            self.pipeline = Compose(self.pipeline)

        self.data_root = data_root
        self.organs = [
            name for name in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, name))
        ]
        self.reduce_zero_label = kwargs.get('reduce_zero_label', False)
        self.label_map = kwargs.get('label_map', None)
    
        self.data_list = self._build_data_list()

        # self.dataset_meta = metainfo

    def __len__(self) -> int:
        # return len(self.organs)
        return len(self.data_list)

    

    def _build_data_list(self) -> List[dict]:
        data_root = self.data_root
        data_list: List[dict] = []
        for organ in self.organs:
            f = os.path.join(data_root, organ)
            f = os.path.join(f, 'imgs')
            for img in os.listdir(f):
                if not img.endswith('.png'):
                    continue
                img_path = os.path.join(f, img)
                seg_map_path = img_path.replace('imgs', 'masks')
                seg_map_path = img_path.replace('img', 'mask') ## for private prostate dataset

                data_info = dict(
                    img_path=img_path,
                    seg_map_path=seg_map_path,
                    is_npy=img_path.endswith('.npy'),
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label,
                    seg_fields=[],
                    # test_mode=test_mode,
                    organ=organ
                )
                data_list.append(data_info)
        return data_list



    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.
        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        # data_info = self.get_data_info(idx)
        data_info = idx
        data_info = self.data_list[idx]
        # print(data_info)
        # return data_info
        return self.pipeline(data_info)

        
    def __getitem__(self, idx: int) -> dict:
        return self.prepare_data(idx)
