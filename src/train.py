from __future__ import print_function
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.append(ROOT.as_posix())
import torch.nn.functional as F
import torch
import torch.nn as nn
from dataset import *
from src.utils import (
    arg_parse, seed_everything, setting, get_logger,
    set_logger, logging_config, backup_envir
)
from src.trainutils import (
    get_model, get_dloaders, get_optim, 
    train, evaluate, test, CrossEntropy
)
        
def main():
    args = arg_parse()
    cfg, device, cur_rank = setting(args)
    writer, timestamp = set_logger(cfg)
    logger = get_logger()
    save_dir = os.path.join(cfg.output_dir, timestamp)
    
    logging_config(cfg)
    seed_everything(cfg.seed)

    # backup evironment
    backup_envir(save_dir)
    logger.info('backup evironment completed !')

    model = get_model(cfg, device)

    vit_patch, vit_patch_idx, vggish, vggish_idx, clip_qst, clip_qst_idx, clip_word, word_mask, clip_word_idx, frame_feat, frame_feat_idx = load_data_into_memory(cfg.data.patch_feat, cfg.data.audio_feat, cfg.data.quest_feat, cfg.data.word_feat, cfg.data.video_feat)

    d_loaders = get_dloaders(cfg, [vit_patch, vit_patch_idx, vggish, vggish_idx, clip_qst, clip_qst_idx, clip_word, word_mask, clip_word_idx, frame_feat, frame_feat_idx])
    optim, sched = get_optim(cfg, model, d_loaders['train'])
    
    best_acc = 0
    best_epoch = -1
    criterion = CrossEntropy()
    
    for epoch in range(1, cfg.epochs + 1):
        logger.info(f"\n-------------- Training epoch {epoch} --------------")
        train(cfg, epoch, device, d_loaders['train'], optim, criterion, model, writer)
        
        logger.info(f"\n-------------- Validation epoch {epoch} --------------")
        acc, loss = evaluate(cfg, epoch, device, d_loaders['val'], criterion, model, writer)
        
        if cfg.hyper_params.sched.name == 'ReduceLROnPlateau':
            if cfg.hyper_params.sched.mode == 'max':
                sched.step(acc)
            elif cfg.hyper_params.sched.mode == 'min':
                sched.step(loss)
        else:
            sched.step(epoch)
        
        if acc >= best_acc and not cfg.debug:
            best_acc = acc
            best_epoch = epoch
            sd = model.module.state_dict()
            new_sd = {}
            for k, v in sd.items():
                if 'video_encoder' not in k:
                    new_sd[k] = v
            
            logger.info(f"best model saved at epoch {epoch} with acc {best_acc}")
            torch.save(new_sd, os.path.join(save_dir, f'best.pt'))
            logger.info(f"-------------- Testing epoch {epoch} --------------")
            test(cfg, device, d_loaders['test'], model)
        
        logger.info(f"Epoch {epoch} done with {acc:3.2f} and loss {loss:.5f}.")
        logger.info(f"At epoch{best_epoch} best acc: {best_acc:3.2f}.")

    if not cfg.debug:
        logger.info(f"\nTesting with Best validation model... {cfg.data.test_annot}")
        cfg.mode = 'test'
        save_dir = Path(save_dir).absolute() 
        best_path = save_dir / f'best.pt'
        original_dict = torch.load(best_path.as_posix())
        update_dict = {}
        for name, param in original_dict.items():
            if hasattr(model, 'module'):
                name = 'module.' +name
            update_dict[name] = param
        model.load_state_dict(update_dict, strict=False)

        test(cfg, device, d_loaders['test'], model)

if __name__ == '__main__':
    main()