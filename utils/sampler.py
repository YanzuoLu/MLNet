"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import itertools
import math
from collections import Counter

import numpy as np
from torch.utils.data.sampler import Sampler


class DistributedWeightedRandomSampler(Sampler):
    def __init__(self, dataset, rank, world_size, seed):
        freq = Counter(dataset.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        self.weights = np.array([class_weight[x] for x in dataset.labels])

        self.rank = rank
        self.world_size = world_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        start = self.rank
        yield from itertools.islice(self._infinite_indices(), start, None, self.world_size)

    def __len__(self):
        return len(self.weights) // self.world_size

    def _infinite_indices(self):
        while True:
            perm = self.rng.choice(len(self.weights), len(self.weights), replace=True, p=self.weights / np.sum(self.weights))
            yield from perm


class DistributedRandomSampler(Sampler):
    def __init__(self, data_source, rank, world_size, seed):
        self.data_source = data_source
        self.rank = rank
        self.world_size = world_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        start = self.rank
        yield from itertools.islice(self._infinite_indices(), start, None, self.world_size)

    def __len__(self):
        return math.ceil(len(self.data_source) / self.world_size)

    def _infinite_indices(self):
        while True:
            perm = self.rng.permutation(len(self.data_source))
            yield from perm


class DistributedSequentialSampler(Sampler):
    def __init__(self, data_source, rank, world_size):
        self.data_source = data_source
        self.rank = rank
        self.world_size = world_size

        shared_size = (len(self.data_source) - 1) // self.world_size + 1
        begin = shared_size * self.rank
        end = min(shared_size * (self.rank + 1), len(self.data_source))
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class CrossSampler(Sampler):
    def __init__(self, source_sampler, target_sampler, padding, batch_size, rank, world_size):
        self.source_sampler = source_sampler
        self.target_sampler = target_sampler
        self.padding = padding
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        self.source_iter = None
        self.target_iter = None

    def __iter__(self):
        source_iter = iter(self.source_sampler)
        target_iter = iter(self.target_sampler)

        while True:
            source_batch_indices = []
            target_batch_indices = []
            for _ in range(self.batch_size):
                source_batch_indices.append(next(source_iter))
                target_batch_indices.append(next(target_iter) + self.padding)

            batch_indices = source_batch_indices + target_batch_indices
            yield from batch_indices

    def __len__(self):
        return len(self.target_sampler) * 2
