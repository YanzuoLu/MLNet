"""
@author: Anonymous Name
@email:  luyz5@mail2.sysu.edu.cn
"""


class InvLRScheduler:
    def __init__(self, optimizer, max_iters, gamma, power):
        self.optimizer = optimizer
        self.base_values = [group['lr'] for group in self.optimizer.param_groups]
        self.max_iters = max_iters
        self.gamma = gamma
        self.power = power

    def step(self, t):
        lrs = [v * (1. + self.gamma * (t / self.max_iters)) ** (-self.power) for v in self.base_values]
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        return lrs
