"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

import utils.scheduler


class Model(nn.Module):
    def __init__(self, num_classes, weights, bias, loss_weights):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weights = loss_weights

        self.base = resnet50(weights=ResNet50_Weights.__dict__[weights])
        self.base = nn.Sequential(*list(self.base.children())[:-1])

        self.cset_fc = nn.Linear(2048, self.num_classes, bias=bias)
        self.oset_fc = nn.Linear(2048, self.num_classes * 2, bias=bias)

    def forward(self, batch):
        imgs, labels = batch['img'], batch['label']

        if not self.training:
            feats = self.base(imgs)
            feats_flatten = torch.flatten(feats, 1)
            cset_logit = self.cset_fc(feats_flatten)
            oset_logit = self.oset_fc(feats_flatten).view(imgs.size(0), 2, self.num_classes)
            oset_prob = F.softmax(oset_logit, dim=1)

            cset_pred = torch.max(cset_logit, dim=1)[1]
            oset_pred = oset_prob[torch.arange(imgs.size(0)), 1, cset_pred] > 0.5
            return cset_pred, oset_pred, feats_flatten

        half_bs = imgs.size(0) // 2
        source_imgs, source_labels = imgs[:half_bs], labels[:half_bs]
        target_imgs = imgs[half_bs:]

        source_feats = self.base(source_imgs)
        source_feats = torch.flatten(source_feats, 1)
        source_cset_logit = self.cset_fc(source_feats)
        source_oset_logit = self.oset_fc(source_feats).view(half_bs, 2, self.num_classes)
        source_oset_prob = F.softmax(source_oset_logit, dim=1)

        source_oset_pos_target = torch.zeros_like(source_cset_logit)
        source_oset_pos_target[torch.arange(half_bs), source_labels] = 1
        source_oset_neg_target = 1 - source_oset_pos_target

        source_cset_loss = F.cross_entropy(source_cset_logit, source_labels)
        source_oset_pos_loss = torch.mean(torch.sum(-source_oset_pos_target * torch.log(source_oset_prob[:,0,:] + 1e-8), dim=1))
        source_oset_neg_loss = torch.mean(torch.max(-source_oset_neg_target * torch.log(source_oset_prob[:,1,:] + 1e-8), dim=1)[0])
        source_oset_loss = source_oset_pos_loss + source_oset_neg_loss

        target_feats = self.base(target_imgs)
        target_feats = torch.flatten(target_feats, 1)
        target_oset_logit = self.oset_fc(target_feats).view(half_bs, 2, self.num_classes)
        target_oset_prob = F.softmax(target_oset_logit, dim=1)
        target_oset_loss = torch.mean(torch.sum(-target_oset_prob * torch.log(target_oset_prob + 1e-8), dim=1))

        loss = source_cset_loss * self.loss_weights['source_cset'] + \
            source_oset_loss * self.loss_weights['source_oset'] + \
            target_oset_loss * self.loss_weights['target_oset']
        metric = {
            'loss': loss.item(),
            'source_cset_loss': source_cset_loss.item(),
            'source_oset_loss': source_oset_loss.item(),
            'target_oset_loss': target_oset_loss.item()
        }
        return loss, metric


def OVANet(cfg, device_id, RANK, WORLD_SIZE):
    model = Model(
        num_classes = cfg.DATASET.SOURCE_PRIVATE + cfg.DATASET.SHARED,
        weights = cfg.MODEL.WEIGHTS,
        bias = cfg.MODEL.BIAS,
        loss_weights = {
            'source_cset': cfg.MODEL.SOURCE_CSET,
            'source_oset': cfg.MODEL.SOURCE_OSET,
            'target_oset': cfg.MODEL.TARGET_OSET
        }
    )

    model.cuda(device_id)
    if WORLD_SIZE > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id],
            find_unused_parameters=cfg.MODEL.FIND_UNUSED_PARAMETERS)

    ft_params = [p for p in model.base.parameters() if p.requires_grad]
    new_params = [p for p in model.cset_fc.parameters() if p.requires_grad] + \
        [p for p in model.oset_fc.parameters() if p.requires_grad]
    param_groups = [{'params': ft_params, 'lr': cfg.OPTIMIZER.FT_LR}, \
        {'params': new_params, 'lr': cfg.OPTIMIZER.NEW_LR}]
    optimizer = torch.optim.SGD(param_groups, momentum=cfg.OPTIMIZER.MOMENTUM, \
        weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    scheduler = utils.scheduler.InvLRScheduler(optimizer, cfg.ENGINE.ITERS, cfg.SCHEDULER.GAMMA, cfg.SCHEDULER.POWER)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.ENGINE.ENABLE_AMP)

    return model, optimizer, scheduler, scaler
