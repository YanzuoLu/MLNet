"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import torch
import torch.distributed as dist
from torch.cuda.amp import custom_bwd, custom_fwd


class GatherLayer(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    @custom_bwd
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out