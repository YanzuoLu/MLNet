"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision.models import ResNet50_Weights, resnet50

import datasets
import utils.sampler
import utils.scheduler
from utils.gather_layer import GatherLayer


class Model(nn.Module):
    def __init__(self, num_classes, num_samples, alpha, neighbor_eps, scale, max_iters, warmup_iters, weights, bias, loss_weights, seed):
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.alpha = alpha
        self.neighbor_eps = neighbor_eps
        self.scale = scale
        self.loss_weights = loss_weights

        self.global_iter = 0
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.rng = np.random.default_rng(seed)

        self.base = resnet50(weights=ResNet50_Weights.__dict__[weights])
        self.base = nn.Sequential(*list(self.base.children())[:-1])
        self.bottleneck = nn.Identity()
        in_features = 2048

        self.cset_fc = nn.Linear(in_features, self.num_classes, bias=bias)
        self.oset_fc = nn.Linear(in_features, self.num_classes * 2, bias=bias)

        self.register_buffer('target_memory_bank', torch.zeros(self.num_samples, 2048))
        self.register_buffer('tt_neighbor_score', torch.zeros(self.num_samples, self.num_samples))

    def forward(self, batch):
        imgs, labels, indices = batch['img'], batch['label'], batch['index']

        if not self.training:
            feats = self.base(imgs)
            feats_flatten = torch.flatten(feats, 1)
            feats_bottleneck = self.bottleneck(feats_flatten)
            cset_logit = self.cset_fc(feats_bottleneck)
            oset_logit = self.oset_fc(feats_bottleneck).view(imgs.size(0), 2, self.num_classes)
            oset_prob = F.softmax(oset_logit, dim=1)

            cset_pred = torch.max(cset_logit, dim=1)[1]
            oset_pred = oset_prob[torch.arange(imgs.size(0)), 1, cset_pred] > 0.5
            return cset_pred, oset_pred, feats_flatten

        half_bs = imgs.size(0) // 2
        source_imgs, target_imgs = imgs[:half_bs], imgs[half_bs:]
        source_labels, target_indices = labels[:half_bs], indices[half_bs:]

        source_feats = self.base(source_imgs)
        source_feats_flatten = torch.flatten(source_feats, 1)
        source_feats_bottleneck = self.bottleneck(source_feats_flatten)
        source_cset_logit = self.cset_fc(source_feats_bottleneck)
        source_oset_logit = self.oset_fc(source_feats_bottleneck).view(half_bs, 2, self.num_classes)
        source_oset_prob = F.softmax(source_oset_logit, dim=1)

        target_feats = self.base(target_imgs)
        target_feats_flatten = torch.flatten(target_feats, 1)
        target_feats_bottleneck = self.bottleneck(target_feats_flatten)
        target_cset_logit = self.cset_fc(target_feats_bottleneck)
        target_cset_prob = F.softmax(target_cset_logit, dim=1)
        target_oset_logit = self.oset_fc(target_feats_bottleneck).view(half_bs, 2, self.num_classes)
        target_oset_prob = F.softmax(target_oset_logit, dim=1)

        if dist.is_initialized() and dist.get_world_size() > 1:
            all_target_indices = torch.cat(GatherLayer.apply(target_indices), dim=0)
            all_target_feats_flatten = torch.cat(GatherLayer.apply(target_feats_flatten), dim=0)
        else:
            all_target_indices = target_indices
            all_target_feats_flatten = target_feats_flatten
        self.target_memory_bank[all_target_indices] = all_target_feats_flatten.float().detach()

        with torch.no_grad():
            if self.global_iter % self.warmup_iters == 0 and self.global_iter != 0:
                target_memory_bank = self.target_memory_bank.cpu()
                tt_sim = F.normalize(target_memory_bank) @ F.normalize(target_memory_bank).T
                tt_sim[torch.arange(self.num_samples), torch.arange(self.num_samples)] = -1.
                tt_nearest = torch.max(tt_sim, dim=1, keepdim=True)[0]
                tt_neighbor_mask = tt_sim > (tt_nearest * self.neighbor_eps)
                mat = tt_neighbor_mask.float()

                ab = mat @ mat.T
                aa = torch.count_nonzero(mat, dim=1).view(-1, 1)
                bb = aa.view(1, -1)
                jaccard_distance = ab / (aa + bb - ab)
                jaccard_distance[torch.arange(self.num_samples), torch.arange(self.num_samples)] = 0

                self.tt_neighbor_score.copy_(jaccard_distance, non_blocking=True)

        source_cset_loss = F.cross_entropy(source_cset_logit, source_labels)
        loss = source_cset_loss * self.loss_weights['source_cset']
        metric = {'source_cset_loss': source_cset_loss.item()}

        if self.loss_weights['source_oset'] > 0.:
            source_oset_pos_target = torch.zeros_like(source_cset_logit)
            source_oset_pos_target[torch.arange(half_bs), source_labels] = 1
            source_oset_pos_loss = torch.mean(torch.sum(-source_oset_pos_target * torch.log(source_oset_prob[:,0,:] + 1e-8), dim=1))

            source_oset_neg_target = 1 - source_oset_pos_target
            source_oset_neg_loss = torch.mean(torch.max(-source_oset_neg_target * torch.log(source_oset_prob[:,1,:] + 1e-8), dim=1)[0])

            source_oset_loss = source_oset_pos_loss + source_oset_neg_loss
            loss += source_oset_loss * self.loss_weights['source_oset']
            metric['source_oset_loss'] = source_oset_loss.item()

        if self.loss_weights['target_oset'] > 0.:
            target_oset_loss = torch.mean(torch.sum(-target_oset_prob * torch.log(target_oset_prob + 1e-8), dim=1))
            loss += target_oset_loss * self.loss_weights['target_oset']
            metric['target_oset_loss'] = target_oset_loss.item()

        if self.loss_weights['mixup'] > 0.:
            mixed_oset_feats_flatten, mixed_oset_target = self.generate_mixed_oset_data(
                imgs, source_labels, source_feats_flatten, target_feats_flatten)
            mixed_oset_feats_bottleneck = self.bottleneck(mixed_oset_feats_flatten)
            mixed_oset_logit = self.oset_fc(mixed_oset_feats_bottleneck).view(half_bs, 2, self.num_classes)
            mixed_oset_prob = F.softmax(mixed_oset_logit, dim=1)

            mixup_loss = torch.mean(torch.sum(-mixed_oset_target * torch.log(mixed_oset_prob[:,1,:] + 1e-8), dim=1))
            loss += mixup_loss * self.loss_weights['mixup']
            metric['mixup_loss'] = mixup_loss.item()

        if self.loss_weights['cc'] > 0.:
            cc_loss = -torch.mean(target_cset_prob * target_oset_prob[:,0,:])
            loss += cc_loss * self.loss_weights['cc']
            metric['cc_loss'] = cc_loss.item()

        if self.loss_weights['neighbor'] > 0.:
            tt_sim = F.normalize(target_feats_flatten) @ F.normalize(self.target_memory_bank).t()
            tt_mask_instance = torch.zeros_like(tt_sim)
            tt_mask_instance[torch.arange(half_bs), target_indices] = 1
            tt_mask_instance = tt_mask_instance.bool()
            tt_sim = (tt_sim + 1.) * (~ tt_mask_instance) - 1.

            tt_nearest = torch.max(tt_sim, dim=1, keepdim=True)[0]
            tt_mask_neighbor = tt_sim > (tt_nearest * self.neighbor_eps)
            tt_num_neighbor = torch.sum(tt_mask_neighbor, dim=1)

            tt_sim_exp = torch.exp(tt_sim * self.scale)
            tt_score = tt_sim_exp / torch.sum(tt_sim_exp, dim=1, keepdim=True)
            tt_neighbor_loss = torch.sum(-torch.log(tt_score + 1e-8) * self.tt_neighbor_score[target_indices] * tt_mask_neighbor, dim=1) / tt_num_neighbor
            neighbor_loss = torch.mean(tt_neighbor_loss)

            if self.global_iter < self.warmup_iters:
                neighbor_loss *= 0.
            loss += neighbor_loss * self.loss_weights['neighbor']
            metric['neighbor_loss'] = neighbor_loss.item()

        self.global_iter += 1
        metric['loss'] = loss.item()
        return loss, metric

    def generate_mixed_oset_data(self, imgs, source_labels, source_feats_flatten, target_feats_flatten):
        half_bs = imgs.size(0) // 2
        mix_factor = torch.tensor(self.rng.beta(self.alpha, self.alpha, half_bs), dtype=torch.float, device='cuda')

        mixed_feats_flatten = mix_factor.view(half_bs, 1) * source_feats_flatten + (1. - mix_factor).view(half_bs, 1) * target_feats_flatten
        mixed_target = torch.zeros(half_bs, self.num_classes, device='cuda')
        mixed_target[torch.arange(half_bs), source_labels] = 1
        return mixed_feats_flatten, mixed_target


def MLNet(cfg, device_id, RANK, WORLD_SIZE):
    train_transform = utils.transforms.__dict__[cfg.DATASET.TRAIN_TRANSFORM]
    target_dataset = datasets.DefaultDataset(cfg.DATASET.TARGET, train_transform)
    target_sampler = utils.sampler.DistributedSequentialSampler(target_dataset, RANK, WORLD_SIZE)
    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size = cfg.ENGINE.EVAL_BATCH_SIZE // WORLD_SIZE,
        sampler = target_sampler,
        num_workers = cfg.ENGINE.NUM_WORKERS // WORLD_SIZE,
        pin_memory = cfg.ENGINE.PIN_MEMORY
    )

    loss_weights = {
        'source_cset': cfg.MODEL.SOURCE_CSET,
        'source_oset': cfg.MODEL.SOURCE_OSET,
        'target_oset': cfg.MODEL.TARGET_OSET,
        'mixup': cfg.MODEL.MIXUP,
        'cc': cfg.MODEL.CC,
        'neighbor': cfg.MODEL.NEIGHBOR
    }
    model = Model(
        num_classes = cfg.DATASET.SOURCE_PRIVATE + cfg.DATASET.SHARED,
        num_samples = len(target_dataset),
        alpha = cfg.MODEL.ALPHA,
        neighbor_eps = cfg.MODEL.NEIGHBOR_EPS,
        scale = cfg.MODEL.SCALE,
        max_iters = cfg.ENGINE.ITERS,
        warmup_iters = cfg.ENGINE.WARMUP_ITERS,
        weights = cfg.MODEL.WEIGHTS,
        bias = cfg.MODEL.BIAS,
        loss_weights = loss_weights,
        seed = cfg.MISC.SEED
    )

    model.cuda(device_id)
    if WORLD_SIZE > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id],
            find_unused_parameters=cfg.MODEL.FIND_UNUSED_PARAMETERS)
    model_without_ddp = model.module if WORLD_SIZE > 1 else model

    model.eval()
    all_feats_flatten = []

    with torch.no_grad():
        for i, batch in enumerate(target_loader):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda(device_id, non_blocking=True)

            with torch.cuda.amp.autocast(cfg.ENGINE.ENABLE_AMP):
                _, _, feats_flatten = model(batch)
            all_feats_flatten.append(feats_flatten.cpu().float())

    all_feats_flatten = torch.cat(all_feats_flatten, dim=0)
    if WORLD_SIZE > 1:
        all_feats_flatten_list = [None for _ in range(WORLD_SIZE)]
        dist.all_gather_object(all_feats_flatten_list, all_feats_flatten)
        all_feats_flatten = torch.cat(all_feats_flatten_list, dim=0)

    model.train()
    model_without_ddp.target_memory_bank.copy_(all_feats_flatten)

    ft_params = [p for p in model.base.parameters() if p.requires_grad]
    new_params = [p for n, p in model.named_parameters() if not n.startswith('base') and p.requires_grad]
    param_groups = [{'params': ft_params, 'lr': cfg.OPTIMIZER.FT_LR}, {'params': new_params, 'lr': cfg.OPTIMIZER.NEW_LR}]

    optimizer = torch.optim.SGD(param_groups, momentum=cfg.OPTIMIZER.MOMENTUM, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    scheduler = utils.scheduler.InvLRScheduler(optimizer, cfg.ENGINE.ITERS, cfg.SCHEDULER.GAMMA, cfg.SCHEDULER.POWER)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.ENGINE.ENABLE_AMP)

    return model, optimizer, scheduler, scaler