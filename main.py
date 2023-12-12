"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import argparse
import builtins
import datetime
import itertools
import math
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.distributed as dist
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import datasets
import models
import utils.sampler
import utils.transforms
from default import _C as cfg
from utils.logger import setup_logger
from utils.metric import AverageMeter


def main():
    device_id = LOCAL_RANK
    torch.cuda.set_device(device_id)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = cfg.ENGINE.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = cfg.ENGINE.ALLOW_TF32
    logger.info(f'set cuda device = {device_id}')

    if WORLD_SIZE > 1:
        logger.info(f'initializing distrubuted training environment')
        dist.init_process_group(backend='nccl')

    logger.info(f'initializing data loader')
    train_loader, eval_loader = initialize_loader()

    logger.info(f'initializing model')
    model, optimizer, scheduler, scaler = models.__dict__[cfg.MODEL.NAME](cfg, device_id, RANK, WORLD_SIZE)
    summary_writer = SummaryWriter(OUTPUT_DIR) if RANK == 0 else None

    model_without_ddp = model.module if WORLD_SIZE > 1 else model
    state = load_checkpoint(model_without_ddp, optimizer)

    hscore, known_acc, unknown_acc, cset_acc = evaluate(model, eval_loader, device_id)
    logger.info(f'[Evaluation Result] H-score: {hscore}, Known Acc: {known_acc}, Unknown Acc: {unknown_acc}, Closed-Set Acc: {cset_acc}')

    start_iter = state.iter + 1
    train_loader_iter = iter(train_loader)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()

    logger.info(f'start training')
    best_metric = 0.

    start_time = time.time()
    end_time = time.time()
    for i in range(start_iter, cfg.ENGINE.ITERS):
        batch = next(train_loader_iter)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(device_id, non_blocking=True)
        data_time.update(time.time() - end_time)

        model.train()
        state.iter = i
        scheduler.step(i)

        with torch.cuda.amp.autocast(cfg.ENGINE.ENABLE_AMP):
            loss, metric = model(batch)

        loss_item = metric['loss']
        if not math.isfinite(loss_item):
            logger.info(f'loss is {loss_item}, stop training')
            sys.exit(1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss.update(loss_item)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % cfg.MISC.PRINT_FREQ == 0 or (i + 1) == cfg.ENGINE.ITERS:
            etas = batch_time.avg * (cfg.ENGINE.ITERS - 1 - i)
            logger.info(
                f'Train ({i+1}/{cfg.ENGINE.ITERS})  '
                f'Time {batch_time.val:.4f}({batch_time.avg:.4f})  '
                f'Data {data_time.val:.4f}({data_time.avg:.4f})  '
                f'Loss {total_loss.val:.4f}({total_loss.avg:.4f})  '
                f'Eta {datetime.timedelta(seconds=int(etas))}'
            )

        if summary_writer:
            for k, v in metric.items():
                summary_writer.add_scalar(k, v, i)

        if (i + 1) % cfg.MISC.EVAL_FREQ == 0:
            hscore, known_acc, unknown_acc, cset_acc = evaluate(model, eval_loader, device_id)
            if summary_writer:
                summary_writer.add_scalar('hscore', hscore, i + 1)
                summary_writer.add_scalar('known_acc', known_acc, i + 1)
                summary_writer.add_scalar('unknown_acc', unknown_acc, i + 1)
                summary_writer.add_scalar('cset_acc', cset_acc, i + 1)
            logger.info(f'[Evaluation Result] H-score: {hscore}, Known Acc: {known_acc}, Unknown Acc: {unknown_acc}, Closed-Set Acc: {cset_acc}')
            save_checkpoint(state, 'checkpoint.pth')
            if cfg.DATASET.TARGET_PRIVATE == 0 and cset_acc > best_metric:
                best_metric = cset_acc
                save_checkpoint(state, 'best_checkpoint.pth')
            elif hscore > best_metric:
                best_metric = hscore
                save_checkpoint(state, 'best_checkpoint.pth')
            logger.info(f'best metric: {best_metric}')

    train_time = time.time() - start_time
    logger.info(f'training completed, running time {datetime.timedelta(seconds=int(train_time))}')


def initialize_loader():
    train_transform = utils.transforms.__dict__[cfg.DATASET.TRAIN_TRANSFORM]
    eval_transform = utils.transforms.__dict__[cfg.DATASET.EVAL_TRANSFORM]

    source_dataset = datasets.DefaultDataset(cfg.DATASET.SOURCE, train_transform)
    target_dataset = datasets.DefaultDataset(cfg.DATASET.TARGET, train_transform)
    eval_dataset = datasets.DefaultDataset(cfg.DATASET.TARGET, eval_transform)

    source_sampler = utils.sampler.DistributedWeightedRandomSampler(source_dataset, RANK, WORLD_SIZE, cfg.MISC.SEED)
    target_sampler = utils.sampler.DistributedRandomSampler(target_dataset, RANK, WORLD_SIZE, cfg.MISC.SEED)

    train_dataset = datasets.CrossDataset(source_dataset, target_dataset)
    train_sampler = utils.sampler.CrossSampler(source_sampler, target_sampler, len(train_dataset.source_dataset), \
        cfg.ENGINE.BATCH_SIZE // WORLD_SIZE, RANK, WORLD_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = cfg.ENGINE.BATCH_SIZE // WORLD_SIZE * 2,
        sampler = train_sampler,
        num_workers = cfg.ENGINE.NUM_WORKERS // WORLD_SIZE,
        pin_memory = cfg.ENGINE.PIN_MEMORY
    )

    eval_sampler = utils.sampler.DistributedSequentialSampler(eval_dataset, RANK, WORLD_SIZE)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size = cfg.ENGINE.EVAL_BATCH_SIZE // WORLD_SIZE,
        sampler = eval_sampler,
        num_workers = cfg.ENGINE.NUM_WORKERS // WORLD_SIZE,
        pin_memory = cfg.ENGINE.PIN_MEMORY
    )

    return train_loader, eval_loader


def evaluate(model, eval_loader, device_id):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()
    num_steps = len(eval_loader)

    all_cset_preds = []
    all_oset_preds = []
    all_labels = []

    start_time = time.time()
    end_time = time.time()
    with torch.no_grad():
        logger.info(f'start evaluating')
        for i, batch in enumerate(eval_loader):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda(device_id, non_blocking=True)
            data_time.update(time.time() - end_time)

            with torch.cuda.amp.autocast(cfg.ENGINE.ENABLE_AMP):
                cset_pred, oset_pred, _ = model(batch)

            all_cset_preds.extend(cset_pred.tolist())
            all_oset_preds.extend(oset_pred.tolist())
            all_labels.extend(batch['label'].tolist())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (i + 1) % cfg.MISC.PRINT_FREQ == 0 or (i + 1) == num_steps:
                etas = batch_time.avg * (num_steps - 1 - i)
                logger.info(
                    f'EVAL ({i+1}/{num_steps})  '
                    f'Time {batch_time.val:.4f}({batch_time.avg:.4f})  '
                    f'Data {data_time.val:.4f}({data_time.avg:.4f})  '
                    f'Eta {datetime.timedelta(seconds=int(etas))}'
                )

    all_cset_preds = np.asarray(all_cset_preds)
    all_oset_preds = np.asarray(all_oset_preds)
    all_labels = np.asarray(all_labels)

    if WORLD_SIZE > 1:
        logger.info(f'gathering results from all GPUs')
        all_cset_preds_list = [None for _ in range(WORLD_SIZE)]
        all_oset_preds_list = [None for _ in range(WORLD_SIZE)]
        all_labels_list = [None for _ in range(WORLD_SIZE)]

        dist.gather_object(all_cset_preds, all_cset_preds_list if RANK == 0 else None, dst=0)
        dist.gather_object(all_oset_preds, all_oset_preds_list if RANK == 0 else None, dst=0)
        dist.gather_object(all_labels, all_labels_list if RANK == 0 else None, dst=0)

        all_cset_preds = np.asarray(list(itertools.chain.from_iterable(all_cset_preds_list))) if RANK == 0 else None
        all_oset_preds = np.asarray(list(itertools.chain.from_iterable(all_oset_preds_list))) if RANK == 0 else None
        all_labels = np.asarray(list(itertools.chain.from_iterable(all_labels_list))) if RANK == 0 else None

    if RANK == 0:
        unknown_class = cfg.DATASET.SHARED + cfg.DATASET.SOURCE_PRIVATE
        unknown_indices = np.where(all_labels == unknown_class)[0]
        known_indices = np.where(all_labels != unknown_class)[0]

        cset_correct = np.where(all_cset_preds[known_indices] == all_labels[known_indices])[0]
        cset_acc = np.mean(np.bincount(all_labels[known_indices][cset_correct], minlength=cfg.DATASET.SHARED) / \
            np.bincount(all_labels[known_indices], minlength=cfg.DATASET.SHARED))

        all_cset_preds[all_oset_preds] = unknown_class
        unknown_correct = np.where(all_cset_preds[unknown_indices] == unknown_class)[0]
        unknown_acc = len(unknown_correct) / len(unknown_indices) if len(unknown_indices) > 0 else 0.
        known_correct = np.where(all_cset_preds[known_indices] == all_labels[known_indices])[0]
        known_acc = np.mean(np.bincount(all_labels[known_indices][known_correct], minlength=cfg.DATASET.SHARED) / \
            np.bincount(all_labels[known_indices], minlength=cfg.DATASET.SHARED))

        hscore = 2. * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-8)
        objects = [hscore, known_acc, unknown_acc, cset_acc]
    else:
        logger.info(f'waiting for evaluation completed...')
        objects = [None, None, None, None]

    if WORLD_SIZE > 1:
        dist.barrier()
        dist.broadcast_object_list(objects, src=0)

    hscore, known_acc, unknown_acc, cset_acc = objects

    eval_time = time.time() - start_time
    logger.info(f'evaluation completed, running time {datetime.timedelta(seconds=int(eval_time))}')
    return hscore, known_acc, unknown_acc, cset_acc


class State:
    def __init__(self, arch, model_without_ddp, optimizer):
        self.arch = arch
        self.iter = -1
        self.model = model_without_ddp
        self.optimizer = optimizer

    def capture_snapshot(self):
        return {
            'arch': self.arch,
            'iter': self.iter,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def apply_snapshot(self, obj):
        msg = self.model.load_state_dict(obj['model'], strict=False)
        if 'arch' in obj.keys() and self.arch == obj['arch']:
            self.iter = obj['iter']
            self.optimizer.load_state_dict(obj['optimizer'])
        return msg

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f):
        snapshot = torch.load(f, map_location='cpu')
        msg = self.apply_snapshot(snapshot)
        logger.info(msg)


def load_checkpoint(model_without_ddp, optimizer):
    state = State(cfg.MODEL.NAME, model_without_ddp, optimizer)
    ckpt_path = os.path.join(OUTPUT_DIR, 'checkpoint.pth')
    if os.path.exists(ckpt_path):
        logger.info(f'loading checkpoint {ckpt_path}')
        state.load(ckpt_path)
    return state


def save_checkpoint(state, ckpt_name):
    if LOCAL_RANK == 0:
        ckpt_path = os.path.join(OUTPUT_DIR, ckpt_name)
        torch.save(state.capture_snapshot(), ckpt_path)
        logger.info(f'saved checkpoint as {ckpt_path}')
    if WORLD_SIZE > 1:
        dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Pytorch Implementation of "MLNet: Mutual Learning Network for Universal Domain Adaptation"''')
    parser.add_argument('--config-file', default='', help='path to config gile', type=str)
    parser.add_argument('opts', default=None, help='modify config using the command-line', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    RANK = int(os.environ['RANK']) if torch.cuda.device_count() > 1 else 0
    LOCAL_RANK = int(os.environ['LOCAL_RANK']) if torch.cuda.device_count() > 1 else 0
    WORLD_SIZE = int(os.environ['WORLD_SIZE']) if torch.cuda.device_count() > 1 else 1

    OUTPUT_DIR = cfg.MISC.OUTPUT_DIR + '_' + cfg.MISC.SUFFIX if cfg.MISC.SUFFIX else cfg.MISC.OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logger(OUTPUT_DIR, LOCAL_RANK, cfg.MODEL.NAME)
    with open(os.path.join(OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    if LOCAL_RANK != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    main()
