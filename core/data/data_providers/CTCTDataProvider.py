# -*- coding: utf-8 -*-
"""

__author__ == Xinzhe Luo
__version__ == 0.1
"""

import glob
import itertools
import logging
import os
# import random
import numpy as np
import torch
import SimpleITK as sitk  # 引入 SimpleITK 库
from torch.utils.data import Dataset
from core.data.image_utils import strsort, load_image_nii
from core.data.intensity_augment import randomIntensityFilter


class DataProvider(Dataset):
    """
    Construct data provider for Task1 Abdomen CT-CT dataset.
    Validation dataset pattern:
    |--valid
    |  |--images
    |  |  |--AbdomenCTCT_0022_0000.nii.gz
    |  |  |--AbdomenCTCT_0023_0000.nii.gz
    |  |  |--...
    |  |--labels
    |  |  |--AbdomenCTCT_0022_0000.nii.gz
    |  |  |--AbdomenCTCT_0023_0000.nii.gz
    |  |  |--...

    """
    dimension = 3

    def __init__(self, data_search_path, training=False, **kwargs):
        self.data_search_path = data_search_path
        self.training = training
        self.kwargs = kwargs
        self.ct_suffix = self.kwargs.pop('ct_suffix', '0000.nii.gz')
        self.ct_range = self.kwargs.pop('ct_range', [-200, 300])
        self.image_prefix = self.kwargs.pop('image_prefix', 'images')
        self.label_prefix = self.kwargs.pop('label_prefix', 'labels')
        self.intensity_aug = self.kwargs.pop('intensity_aug', False)
        self.equalize_hist = self.kwargs.pop('equalize_hist', False)
        self.crop_shape = np.asarray(self.kwargs.pop('crop_shape', (112, 96, 112)),
                                     dtype=np.int32)

        self.data_pair_names = self._find_data_names(self.data_search_path)

    def __len__(self):
        return len(self.data_pair_names)

    def get_image_name(self, index):
        return self.data_pair_names[index]

    def _find_data_names(self, data_search_path):
        """
        Get pairs of image names.

        :param data_search_path:
        :return:
        """
        all_nii_names = strsort(glob.glob(os.path.join(data_search_path,
                                                       '**/*.nii.gz'),
                                          recursive=True))
        all_nii_names = [os.path.normpath(name) for name in all_nii_names]
        all_img_names = [name for name in all_nii_names if name.split(os.path.sep)[-2] == self.image_prefix]

        CT_img_names = [
            name for name in all_img_names if self.ct_suffix in os.path.basename(name)
        ]

        if self.training:
            pair_names = list(itertools.product(CT_img_names, CT_img_names))
        else:
            pair_names = list(itertools.permutations(CT_img_names, r=2))

        return pair_names

    def resample_image(self, image, new_spacing):
        """
        重采样图像到指定的间距
        :param image: SimpleITK 图像对象
        :param new_spacing: 新的图像间距
        :return: 重采样后的 SimpleITK 图像对象
        """
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in
                    zip(original_size, original_spacing, new_spacing)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resampler.SetInterpolator(sitk.sitkBSpline)
        return resampler.Execute(image)

    def __getitem__(self, item):
        pair_names = self.data_pair_names[item]
        name1, name2 = pair_names

        img1_itk = sitk.ReadImage(name1)
        img2_itk = sitk.ReadImage(name2)

        new_spacing = [3, 3, 3]  # 固定为 [3, 3, 3] 的间距
        img1_itk = self.resample_image(img1_itk, new_spacing)
        img2_itk = self.resample_image(img2_itk, new_spacing)

        img1 = sitk.GetArrayFromImage(img1_itk)
        img2 = sitk.GetArrayFromImage(img2_itk)

        img1 = np.clip(img1, a_min=self.ct_range[0], a_max=self.ct_range[1])
        img2 = np.clip(img2, a_min=self.ct_range[0], a_max=self.ct_range[1])

        if np.min(img1) < 0:
            img1 = img1 - np.min(img1)
        if np.min(img2) < 0:
            img2 = img2 - np.min(img2)

        if self.intensity_aug:
            img1 = randomIntensityFilter(img1)
            img2 = randomIntensityFilter(img2)

        lab1 = load_image_nii(name1.replace(self.image_prefix, self.label_prefix))[0]
        lab2 = load_image_nii(name2.replace(self.image_prefix, self.label_prefix))[0]

        images = np.stack([img1, img2])  # [2, *vol_shape]
        labels = np.stack([lab1, lab2])  # [2, *vol_shape]

        # crop roi
        half = np.asarray(images.shape[-self.dimension:]) // 2
        r = self.crop_shape // 2
        images = images[..., half[0] - r[0]:half[0] + r[0],
                 half[1] - r[1]:half[1] + r[1], half[2] - r[2]:half[2] + r[2]]
        labels = labels[..., half[0] - r[0]:half[0] + r[0],
                 half[1] - r[1]:half[1] + r[1], half[2] - r[2]:half[2] + r[2]]

        names = [os.path.basename(name)[:-7] for name in pair_names]
        names = '_'.join(names)

        return {
            'images': images,
            'labels': labels,
            'affines': [img1_itk.GetDirection(), img2_itk.GetDirection()],
            'headers': [img1_itk.GetMetaDataDictionary(), img2_itk.GetMetaDataDictionary()],
            'names': names,
        }

    def data_collate_fn(self, batch):
        AF = [data.pop('affines') for data in batch]
        HE = [data.pop('headers') for data in batch]
        batch_tensor = dict([(k, torch.stack([torch.from_numpy(data[k]) for data in batch])) for k in batch[0].keys()])
        batch_tensor['affines'] = AF
        batch_tensor['headers'] = HE

        return batch_tensor