import torch
from core.data.data_providers.OASISDataProvider import DataProvider
import os
import numpy as np
from core.data.image_utils import get_label_center


class OASIS3D:
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = 'valid' if mode == 'val' else mode
        self.modalities = cfg.dataset.mods

        root_path = '/content/drive/MyDrive/OASIS'

        if mode == 'train':
            training = True
            oasis_max_length = 100 #默认会限制最多加载 10000 个样本对。修改为200 减少训练耗时，同时保留多样性
        else:
            training = False
            oasis_max_length = 200

        if mode == 'val' or self.cfg.dataset.one_sample_only:
            data_search_path = f'{root_path}/valid'
        elif mode == 'train':
            data_search_path = f'{root_path}/train'
        elif mode == 'test':
            data_search_path = f'{root_path}/test'

        self.dataset = DataProvider(data_search_path,
                                    training=training, max_length=oasis_max_length,
                                    intensity_aug=cfg.dataset.intensity_aug & (self.mode == 'train'))

    def __len__(self):
        if self.mode == 'test' and self.cfg.exp.test.save_result.enable and self.cfg.exp.test.save_result.idx_sample >= 0:
            return 1
        if self.cfg.dataset.one_sample_only:
            if self.mode == 'train':
                return 500
            else:
                return 1
        else:
            return len(self.dataset)

    def __getitem__(self, item):
        if self.mode == 'test' and self.cfg.exp.test.save_result.enable and self.cfg.exp.test.save_result.idx_sample >= 0:
            item = self.cfg.exp.test.save_result.idx_sample
        if self.cfg.dataset.one_sample_only:
            return self.dataset[0]
        else:
            return self.dataset[item]

    def get_batch(self, samples):
        batch = {}

        imgs = torch.stack([torch.from_numpy(sample.pop('images')) for sample in samples]) # [B, 2, *vol_shape]
        if self.cfg.dataset.normalization == 'min-max':
            imgs = imgs - torch.amin(imgs, dim=list(range(2, len(imgs.shape))), keepdim=True)
            imgs = imgs / torch.amax(imgs, dim=list(range(2, len(imgs.shape))), keepdim=True)
        elif self.cfg.dataset.normalization == 'z-score':
            imgs = imgs - torch.mean(imgs, dim=list(range(2, len(imgs.shape))), keepdim=True)
            imgs = imgs / torch.std(imgs, dim=list(range(2, len(imgs.shape))), keepdim=True)
        else:
            raise NotImplementedError

        segs = torch.stack([torch.from_numpy(sample.pop('labels')) for sample in samples]) # [B, 2, *vol_shape]
        masks = torch.stack([torch.from_numpy(sample.pop('masks')) for sample in samples]) # [B, 2, *vol_shape]

        names = [data.pop('names') for data in samples]
        batch['imgs'] = imgs
        batch['segs'] = segs
        batch['masks'] = masks
        batch['names'] = names

        return batch

    def to_device(self, batch, device):
        imgs = batch['imgs'].to(device)
        segs = batch['segs'].to(device)
        masks = batch['masks'].to(device)
        names = batch['names']

        # imgs: [B, 2, H, W, D]
        # segs: [B, 2, H, W, D]
        # masks: [B, 2, H, W, D]
        data = {'imgs': imgs, 'masks': masks, 'segs': segs, 'names': names}
        # data = {'imgs': imgs, 'segs': segs}

        if self.cfg.model.tre.label_center:
            keypoints = [get_label_center(seg) * 2 for seg in segs] # B * [2, n, 3
            data['keypoints'] = keypoints

        return data
