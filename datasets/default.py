"""
@author: Anonymous Name
@email:  Anonymous Email
"""

from PIL import Image
from torch.utils.data import Dataset


class DefaultDataset(Dataset):
    def __init__(self, root, transform):
        self.img_paths = []
        self.labels = []
        self.transform = transform

        with open(root) as f:
            for x in f.readlines():
                img_path, label = x.strip().split(' ')
                self.img_paths.append(img_path)
                self.labels.append(int(label))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return {
            'index': index,
            'img': img,
            'label': label
        }

    def __len__(self):
        return len(self.img_paths)