import glob
import os
import random
from typing import Dict

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_a = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_b = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_b = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_a, 'B': item_b}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class BiDataset(Dataset):
    KEY_DATASET: str = 'dataset'
    KEY_IM_KEY: str = 'im_key'

    @staticmethod
    def opt(dataset: Dataset, key: str):
        return {
            BiDataset.KEY_DATASET: dataset,
            BiDataset.KEY_IM_KEY: key
        }

    def __init__(self, dataset_source_opt: Dict, dataset_target_opt: Dict, transform, unaligned=True):
        self._dataset_source = dataset_source_opt[BiDataset.KEY_DATASET]
        self._dataset_target = dataset_target_opt[BiDataset.KEY_DATASET]
        self._source_key = dataset_source_opt[BiDataset.KEY_IM_KEY]
        self._target_key = dataset_target_opt[BiDataset.KEY_IM_KEY]
        self.transform = transform
        self.unaligned = unaligned

    def __getitem__(self, index):
        item_a = self._dataset_source[index % len(self._dataset_source)]

        if self.unaligned:
            item_b = self._dataset_target[random.randint(0, len(self._dataset_target) - 1)]
        else:
            item_b = self._dataset_target[index % len(self._dataset_target)]
        if self.transform is not None:
            item_a = self.transform(item_a)
            item_b = self.transform(item_b)
        return {'A': item_a, 'B': item_b}

    def __len__(self):
        return max(len(self._dataset_source), len(self._dataset_target))
