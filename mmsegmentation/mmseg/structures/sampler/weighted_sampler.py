# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import compute_sample_weight
from pytorch_toolbelt.utils import fs
from torch.utils.data import WeightedRandomSampler
import numpy as np
from .base_pixel_sampler import BasePixelSampler
import itertools
import math
from typing import Iterator, Optional, Sized
import os
import torch
import skimage
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS

from torch.utils.data import Dataset as BaseDataset

@DATA_SAMPLERS.register_module()
class WeightedSampler(Sampler):
    """Online Hard Example Mining Sampler for segmentation.

    Args:
        ds (Dataset): The dataset.
        mul_factor (float, optional)
    """

    def __init__(self,
                 images_dir, 
                 masks_dir,
                 dataset: Sized,
                 mul_factor=5,
                 shuffle:bool=False,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rsdataset = RSDataset(images_dir=images_dir, masks_dir=masks_dir)
        self.weights = self.get_balanced_weights(rsdataset)
        self.sampler = WeightedRandomSampler(self.weights, len(rsdataset) * mul_factor)

    def get_balanced_weights(self, dataset):
        labels=[]
        for mask in dataset.images_fps:
            mask = fs.read_image_as_is(mask)
            unique_labels = np.unique(mask)
            labels.append(''.join([str(int(i)) for i in unique_labels]))

        weights = compute_sample_weight('balanced', labels)
        return weights
        
    def __iter__(self) -> Iterator[int]:
        return self.sampler.__iter__()

    def __len__(self) -> int:
        return self.sampler.__len__()
    
    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class RSDataset(BaseDataset):

    def __init__(
            self,
            images_dir,
            masks_dir
    ):
        self.ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)

        self.ids.sort()
        self.mask_ids.sort()

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

        self.class_values = np.arange(0,6,1)

    @staticmethod
    def _read_img(image_path):
        img = skimage.io.imread(image_path, plugin='tifffile')
        return img

    def __getitem__(self, i):

        image = self._read_img(self.images_fps[i])
        image = image[:,:,0:3]
        mask =  self._read_img(self.masks_fps[i])
        image = image.transpose(2, 0, 1).astype('float32')
        mask = mask.transpose(2, 0, 1).astype('float32')
        image = torch.as_tensor(image, dtype=torch.float32).cuda()
        mask = torch.as_tensor(mask, dtype=torch.float32).cuda()


        return image, mask

    def __len__(self):
        return len(self.ids)