"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

from torch.utils.data import Dataset

from .default import DefaultDataset


class CrossDataset(Dataset):
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        return len(self.source_dataset) + len(self.target_dataset)

    def __getitem__(self, index):
        if index < len(self.source_dataset):
            return self.source_dataset[index]
        else:
            return self.target_dataset[index - len(self.source_dataset)]
