from typing import Dict, List, Tuple
from collections import defaultdict

import json
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils import get_logger
from src.dataset import AVQA_dataset, qtype2idx
from src.utils import calculate_parameters
from src.models.net import STG
from dataset import *


class AverageMeter(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
    
    def reset(self):
        self.values = defaultdict(float)
        self.count  = 0

    def update(self, val: List[Tuple[str, float]], step_n: int):
        for key, val in val:
            self.values[key] += val
        self.count += step_n

    def get(self, key: str):
        return self.values[key] / self.count

class CrossEntropy(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction='mean'):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)  # (B, C)
        n_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = - (true_dist * log_probs).sum(dim=1)  # (B,)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
def get_model(cfg: dict,
              device: torch.device):
    hyper_params = cfg.hyper_params 
    
    if hyper_params.model_type.startswith('STG'):
        model = STG(**hyper_params.model)
    else:
        raise NotImplementedError(f"Model type {hyper_params.model_type} is not implemented")
    
    model = model.to(device)
    if cfg.weight is not None and cfg.weight != '':
        logger = get_logger()
        
        weight = cfg.weight
        msg = model.load_state_dict(torch.load(weight), strict=False)
        logger.info(f'Missing keys: {json.dumps(msg.missing_keys, indent=4)}')
        logger.info(f'Unexpected keys: {json.dumps(msg.unexpected_keys, indent=4)}')
        logger.info(f"=> loaded successfully '{weight}'")
    
    model = nn.DataParallel(model)
    
    if cfg.mode == 'train':
        calculate_parameters(model)
    
    return model


def get_optim(cfg: dict,
              model: nn.Module,
              train_loader: DataLoader):
    logger = get_logger()
    
    if cfg.hyper_params.optim.encoder_lr is not None:
        m = model.module if hasattr(model, 'module') else model
        
        other_params = [
            param for name, param in model.named_parameters() \
            if 'video_encoder' not in name and 'quest_encoder' not in name and \
                'audio_encoder' not in name and 'mllm' not in name
        ]
        encoder_params = [
            param for name, param in model.named_parameters() \
            if 'video_encoder' in name or 'quest_encoder' in name or \
                'audio_encoder' in name or 'mllm' in name
        ]
        params = [
            {'params': other_params, 'lr': cfg.hyper_params.optim.lr},
            {'params': encoder_params, 'lr': cfg.hyper_params.optim.encoder_lr},
        ]
    else:
        params = model.parameters()
    
    optimizer = optim.AdamW(params, lr=cfg.hyper_params.optim.lr,
                           weight_decay=cfg.hyper_params.optim.weight_decay,
                           betas=cfg.hyper_params.optim.betas)
    
    for param_group in optimizer.param_groups:
        logger.info("\n-------------- optimizer info --------------")
        logger.info(f'Learning rate: {param_group["lr"]}')
        logger.info(f'Betas: {param_group["betas"]}')
        logger.info(f'Eps: {param_group["eps"]}')
        logger.info(f'Weight decay: {param_group["weight_decay"]}')

    if 'StepLR' in cfg.hyper_params.sched.name:
        milestone = [cfg.hyper_params.sched.decay_start + cfg.hyper_params.sched.decay_every * i 
                     for i in range((cfg.epochs - cfg.hyper_params.sched.decay_start) // cfg.hyper_params.sched.decay_every)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=cfg.hyper_params.sched.gamma)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.hyper_params.sched.mode,
            factor=cfg.hyper_params.sched.factor,
            patience=cfg.hyper_params.sched.patience,
            verbose=cfg.hyper_params.sched.verbose)
    
    return optimizer, scheduler


def get_dloaders(cfg: dict, data) -> Dict[str, DataLoader]:
    hyper_params = cfg.data

    vit_patch, vit_patch_idx, vggish, vggish_idx, clip_qst, clip_qst_idx, clip_word, word_mask, clip_word_idx, frame_feat, frame_feat_idx = data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10]

    train_dataset = AVQA_dataset(cfg, mode=cfg.mode,
                                 audios_input = [vggish, vggish_idx], 
                                 clip_input = [vit_patch, vit_patch_idx],
                                 clip_qst_input = [clip_qst, clip_qst_idx], 
                                 clip_word_input = [clip_word, word_mask, clip_word_idx], 
                                 frame_feat_input = [frame_feat, frame_feat_idx]
                                 )
    val_dataset = AVQA_dataset(cfg, mode='valid',
                                 audios_input = [vggish, vggish_idx], 
                                 clip_input = [vit_patch, vit_patch_idx],
                                 clip_qst_input = [clip_qst, clip_qst_idx], 
                                 clip_word_input = [clip_word, word_mask, clip_word_idx], 
                                 frame_feat_input = [frame_feat, frame_feat_idx]
                                 )
    
    test_dataset = AVQA_dataset(cfg, mode='test',
                                 audios_input = [vggish, vggish_idx], 
                                 clip_input = [vit_patch, vit_patch_idx],
                                 clip_qst_input = [clip_qst, clip_qst_idx], 
                                 clip_word_input = [clip_word, word_mask, clip_word_idx], 
                                 frame_feat_input = [frame_feat, frame_feat_idx]
                                 )
    
    train_sampler = None
    valid_sampler = None
    b_size = hyper_params.batch_size
        
    train_loader = DataLoader(train_dataset, 
                                batch_size=b_size,
                                shuffle=(train_sampler is None) and (cfg.mode == 'train'), 
                                num_workers=hyper_params.num_workers,
                                pin_memory=True,
                                worker_init_fn=None,
                                sampler=train_sampler)
    val_loader = DataLoader(val_dataset,
                            batch_size=hyper_params.eval_batch_size,
                            shuffle=False,
                            num_workers=hyper_params.num_workers,
                            sampler=valid_sampler,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=hyper_params.eval_batch_size,
                            shuffle=False,
                            num_workers=hyper_params.num_workers,
                            sampler=valid_sampler,
                            pin_memory=True)
    return {
        f"{cfg.mode}": train_loader,
        'val': val_loader,
        'test':test_loader,
    }


def get_items(batch: dict, device: torch.device):
    reshaped_data = dict(
        quest=batch['quest'].to(device).float(),
        audio=batch['audio'].to(device).float(),
        video=batch['video'].to(device).float(),
        word=batch['word'].to(device).float(),
        word_mask=batch['word_mask'].to(device),
        qtype_label=batch['qtype_label'].to(device).long(),
        patch=batch['patch'].to(device).float() if 'patch' in batch else None,
        n_quest=batch['n_quest'] if 'n_quest' in batch else None,
        n_video=batch['n_video'].to(device).float() if 'n_video' in batch else None,
        n_audio=batch['n_audio'].to(device).float() if 'n_audio' in batch else None,
        n_patch=batch['n_patch'].to(device).float() if 'n_patch' in batch else None,
        prompt=batch['prompt'] if 'prompt' in batch else None,
        label=batch['label'].to(device).long().reshape(-1, ),
    )
    
    reshaped_data['quest'] = reshaped_data['quest'].to(device)
    if reshaped_data['n_quest'] is not None:
        reshaped_data['n_quest'] = reshaped_data['n_quest'].to(device)

    return reshaped_data

def train(cfg: dict,
          epoch: int,
          device: torch.device,
          train_loader: DataLoader,
          optimizer: Optimizer,
          criterion: nn.Module,
          model: nn.Module,
          writer: None,
    ):
    logger = get_logger()
    
    model.train()
    avg_meter = AverageMeter()
    tot_batch = len(train_loader) - 1
    time_cost = 0
    epoch_time = time.time()
    for batch_idx, sample in enumerate(train_loader):
        start_time = time.time()

        reshaped_data = get_items(sample, device)
        optimizer.zero_grad()
        start = time.time()
        output = model(reshaped_data)
        
        loss = 0
        target = reshaped_data['label']
        ce_loss = criterion(output['out'], target)
        loss += ce_loss
        losses = [('ce_loss', ce_loss)]
        for key in output:
            if 'loss' in key:
                losses.append((key, output[key]))
                loss += output[key]
        losses.append(('total_loss', loss))
        loss.backward()
        optimizer.step()

        avg_meter.update(losses, step_n=1)

        end = time.time()
        time_cost += (end - start) * 1000 
        
        if batch_idx % cfg.log_interval == 0 or batch_idx == len(train_loader) - 1:
            batch_t = time.time() - start_time 
            elapsed_t = time.time() - epoch_time
            avg_time = elapsed_t / (batch_idx + 1)
            est_time = (tot_batch - batch_idx - 1) * avg_time / 60

            cur_batch = str(batch_idx).zfill(len(str(tot_batch)))
            batch_ratio = 100. * batch_idx / tot_batch
            log_string = (
                f'[EST: {est_time:7.2f}m][Process Time: {batch_t:7.2f}s]'
                f'- Epoch: {epoch} [{cur_batch}/{tot_batch} ({batch_ratio:3.0f}%)]'
                '\tLosses: '
            )
            loss_string = (
                ' '.join([f'{key}-{value:.4f}({avg_meter.get(key):.4f})' for key, value in losses])
            )
            logger.info(msg=log_string + loss_string)
        
        if cfg.debug and batch_idx == 10:
            break

    time_cost = time_cost / (batch_idx+1)
    print(f"Train time: {time_cost:.4f} ms")


def evaluate(cfg: dict,
            epoch: int,
            device: torch.device,
            val_loader: DataLoader,
            criterion: nn.Module,
            model: nn.Module,
            writer:  None):
    global qtype2idx
    
    logger = get_logger()
    model.eval()
    
    loss = 0
    total, correct = 0, 0
    tot_tensor = torch.zeros(9, dtype=torch.long).to(device)
    correct_tensor = torch.zeros(9, dtype=torch.long).to(device)
    time_cost = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            reshaped_data = get_items(sample, device)
            qst_types = sample['type']
            target = reshaped_data['label']
            output = model(reshaped_data)
            start = time.time()
            _, predicted = torch.max(output['out'].data, 1)
            end = time.time()
            time_cost += (end - start) * 1000 

            total += predicted.size(0)
            correct += (predicted == target).sum().item()
            loss += criterion(output['out'], target) / len(val_loader)
            for idx, (modal_type, qst_type) in enumerate(zip(qst_types[0], qst_types[1])):
                gather_idx = qtype2idx[modal_type][qst_type]
                tot_tensor[gather_idx] += 1
                correct_tensor[gather_idx] += (predicted[idx] == target[idx]).long().item()
        
            if cfg.debug and batch_idx == 10:
                break
        
            if batch_idx % cfg.log_interval == 0 or batch_idx == len(val_loader) - 1:
                logger.info(f'Test progress: {batch_idx:3.0f}/{len(val_loader) - 1}')
    
    acc = correct / total * 100.
    loss = loss.item()

    for modality in ['Audio', 'Visual', 'Audio-Visual']:
        modality_corr = 0
        modality_tot = 0
        
        for qst_type in qtype2idx[modality]:
            corr = correct_tensor[qtype2idx[modality][qst_type]].item()
            tot = tot_tensor[qtype2idx[modality][qst_type]].item()
            
            modality_corr += corr
            modality_tot += tot
            value = corr / tot * 100.
            
            key = f'{modality}/{qst_type}'
            logger.info(f'Epoch {epoch} - {key:>24} accuracy: {value:.2f}({corr}/{tot})')
        
        modality_acc = modality_corr / modality_tot * 100.
        logger.info(f'Epoch {epoch} - {modality:>24} accuracy: {modality_acc:.2f}({modality_corr}/{modality_tot})')
    
    key = 'Total'
    logger.info(f'Epoch {epoch} - {key:>24} accuracy: {acc:.2f}({correct}/{total})')
    time_cost = time_cost / (batch_idx+1)
    print(f"Val time: {time_cost:.4f} ms")
    return acc, loss


def test(cfg: dict,
         device: torch.device,
         val_loader: DataLoader,
         model: nn.Module):
    global qtype2idx
    
    logger = get_logger()
    model.eval()

    total, correct = 0, 0
    tot_tensor = torch.zeros(9, dtype=torch.long).to(device)
    correct_tensor = torch.zeros(9, dtype=torch.long).to(device)
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            reshaped_data = get_items(sample, device)

            qst_types = sample['type']
            target = reshaped_data['label']
            output = model(reshaped_data)
            _, predicted = torch.max(output['out'].data, 1)
            total += predicted.size(0)
            correct += (predicted == target).sum().item()
            for idx, (modal_type, qst_type) in enumerate(zip(qst_types[0], qst_types[1])):
                gather_idx = qtype2idx[modal_type][qst_type]
                tot_tensor[gather_idx] += 1
                correct_tensor[gather_idx] += (predicted[idx] == target[idx]).long().item()

            if cfg.debug and batch_idx == 10:
                break

            if batch_idx % cfg.log_interval == 0 or batch_idx == len(val_loader) - 1:
                logger.info(f'Test progress: {batch_idx:3.0f}/{len(val_loader) - 1}')

    acc = correct / total * 100.
    for modality in ['Audio', 'Visual', 'Audio-Visual']:
        modality_corr = 0
        modality_tot = 0
        
        for qst_type in qtype2idx[modality]:
            corr = correct_tensor[qtype2idx[modality][qst_type]].item()
            tot = tot_tensor[qtype2idx[modality][qst_type]].item()
            
            modality_corr += corr
            modality_tot += tot
            value = corr / tot * 100.
            
            key = f'{modality}/{qst_type}'
            logger.info(f'Test {key:>24} accuracy: {value:.2f}({corr}/{tot})')
        
        modality_acc = modality_corr / modality_tot * 100.
        logger.info(f'Test {modality:>24} accuracy: {modality_acc:.2f}({modality_corr}/{modality_tot})')
    key = 'Total avg'
    logger.info(f'Test {key:>24} accuracy: {acc:.2f}({correct}/{total})')
    return acc
