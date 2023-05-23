"""
@author: Anonymous Name
@email:  Anonymous Email
"""

from PIL import Image
from torch.utils.data import Dataset


class DefaultDataset(Dataset):
    def __init__(self, root, transform):
        self.imgs = []
        self.labels = []
        self.transform = transform

        with open(root) as f:
            for x in f.readlines():
                img_path, label = x.strip().split(' ')
                self.imgs.append(Image.open(img_path).convert('RGB'))
                self.labels.append(int(label))

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index]
        img = self.transform(img)
        return {
            'index': index,
            'img': img,
            'label': label
        }

    def __len__(self):
        return len(self.imgs)