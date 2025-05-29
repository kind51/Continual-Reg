# -*- coding: utf-8 -*-
"""
OASIS 数据集数据提供器（含重采样）
"""
import glob
import itertools
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
from core.data.image_utils import strsort, load_image_nii  # 假设存在该工具函数
from core.data.intensity_augment import randomIntensityFilter


class DataProvider(Dataset):
    dimension = 3

    def __init__(self, data_search_path, training=True, **kwargs):
        self.data_search_path = data_search_path
        self.training = training
        self.kwargs = kwargs
        self.mr_suffix = kwargs.pop('mr_suffix', '0000.nii.gz')
        self.mr_range = kwargs.pop('mr_range', [-np.inf, np.inf])
        self.image_prefix = kwargs.pop('image_prefix', 'images')
        self.label_prefix = kwargs.pop('label_prefix', 'labels')
        self.mask_prefix = kwargs.pop('mask_prefix', 'masks')
        self.intensity_aug = kwargs.pop('intensity_aug', False)
        self.pad_shape = np.asarray(kwargs.pop('crop_shape', (112, 96, 112)), dtype=np.int32)
        self.max_length = kwargs.pop('max_length', 10000)  # 控制数据集大小

        # 查找所有图像对
        self.data_pair_names = self._find_data_names(data_search_path)

    def __len__(self):
        return min(len(self.data_pair_names), self.max_length)

    def _find_data_names(self, data_search_path):
        """获取所有图像对路径"""
        all_nii_names = strsort(glob.glob(os.path.join(data_search_path, '**/*.nii.gz'), recursive=True))
        all_img_names = [name for name in all_nii_names if self.image_prefix in name]
        mr_img_names = [name for name in all_img_names if self.mr_suffix in os.path.basename(name)]

        # 生成训练/验证对（训练集为笛卡尔积，验证集为排列）
        if self.training:
            pair_names = list(itertools.product(mr_img_names, mr_img_names))
        else:
            pair_names = list(itertools.permutations(mr_img_names, 2))
        return pair_names

    def resample_image(self, image_path, new_spacing):
        """重采样图像到指定间距"""
        img_itk = sitk.ReadImage(image_path)
        original_spacing = img_itk.GetSpacing()
        original_size = img_itk.GetSize()

        # 计算新尺寸
        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in
                    zip(original_size, original_spacing, new_spacing)]

        # 重采样设置
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(img_itk.GetDirection())
        resampler.SetOutputOrigin(img_itk.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(img_itk.GetPixelIDValue())
        resampler.SetInterpolator(sitk.sitkBSpline)  # 插值方式
        return resampler.Execute(img_itk), img_itk.GetDirection()  # 返回重采样后的图像和仿射矩阵

    def __getitem__(self, item):
        name1, name2 = self.data_pair_names[item]

        # OASIS 重采样到 2x2x2 mm
        new_spacing = [2, 2, 2] if self.training else [2, 2, 2]  # 训练/验证均用 2x2x2
        img1_itk, aff1 = self.resample_image(name1, new_spacing)
        img2_itk, aff2 = self.resample_image(name2, new_spacing)

        # 转换为 numpy 数组
        img1 = sitk.GetArrayFromImage(img1_itk)
        img2 = sitk.GetArrayFromImage(img2_itk)

        # 强度裁剪（MR 通常不需要固定范围，使用百分位裁剪）
        img1 = np.clip(img1, a_min=np.percentile(img1, 1), a_max=np.percentile(img1, 99))
        img2 = np.clip(img2, a_min=np.percentile(img2, 1), a_max=np.percentile(img2, 99))

        # 强度增强（可选）
        if self.intensity_aug:
            img1 = randomIntensityFilter(img1)
            img2 = randomIntensityFilter(img2)

        # 加载标签和掩码
        lab1 = load_image_nii(name1.replace(self.image_prefix, self.label_prefix))[0]
        lab2 = load_image_nii(name2.replace(self.image_prefix, self.label_prefix))[0]
        mask1 = load_image_nii(name1.replace(self.image_prefix, self.mask_prefix))[0]
        mask2 = load_image_nii(name2.replace(self.image_prefix, self.mask_prefix))[0]

        # 调整维度顺序：[C, D, H, W]
        images = np.stack([img1, img2]).transpose((0, 3, 1, 2))  # [2, D, H, W]
        labels = np.stack([lab1, lab2]).transpose((0, 3, 1, 2))
        masks = np.stack([mask1, mask2]).transpose((0, 3, 1, 2))

        # 填充到固定尺寸 (112, 96, 112)
        current_shape = np.array(images.shape[1:])  # [D, H, W]
        pad_width = [(0, 0)] + [((self.pad_shape[i] - current_shape[i]) // 2,
                                 self.pad_shape[i] - current_shape[i] - (self.pad_shape[i] - current_shape[i]) // 2)
                                for i in range(3)]
        images = np.pad(images, pad_width, mode='constant')
        labels = np.pad(labels, pad_width, mode='constant')
        masks = np.pad(masks, pad_width, mode='constant')

        # 获取头信息（假设 load_image_nii 可返回头信息，或从 ITK 图像获取）
        head1 = img1_itk.GetMetaDataDictionary()  # 若 SimpleITK 图像有元数据
        head2 = img2_itk.GetMetaDataDictionary()

        return {
            'images': images.astype(np.float32),
            'labels': labels.astype(np.float32),
            'masks': masks.astype(np.float32),
            'affines': [aff1, aff2],
            'headers': [head1, head2],
            'names': f"{os.path.basename(name1)[:-7]}_{os.path.basename(name2)[:-7]}"
        }

    def data_collate_fn(self, batch):
        """数据整理函数"""
        affines = [data.pop('affines') for data in batch]
        headers = [data.pop('headers') for data in batch]
        batch_tensor = {
            k: torch.from_numpy(np.stack([data[k] for data in batch])).float()
            for k in batch[0].keys() if k not in ['affines', 'headers', 'names']
        }
        batch_tensor['affines'] = affines
        batch_tensor['headers'] = headers
        batch_tensor['names'] = [data['names'] for data in batch]
        return batch_tensor