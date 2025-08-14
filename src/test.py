from __future__ import print_function
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.append(ROOT.as_posix())
from dataset import *
import torch
import torch.nn as nn

from src.utils import (
    arg_parse, seed_everything, setting, get_logger,
    set_logger, logging_config
)
from src.trainutils import (
    get_model, get_dloaders, test
)

def main():
    args = arg_parse()
    args.mode = 'test'
    args.weight = './save/best.pt'
    args.output_path = './save/'
    cfg, device, cur_rank = setting(args)
    set_logger(cfg)
    logger = get_logger()
    
    logging_config(cfg)
    seed_everything(cfg.seed)
    vit_patch, vit_patch_idx, vggish, vggish_idx, clip_qst, clip_qst_idx, clip_word, word_mask, clip_word_idx, frame_feat, frame_feat_idx = load_data_into_memory(cfg.data.patch_feat, cfg.data.audio_feat, cfg.data.quest_feat, cfg.data.word_feat, cfg.data.video_feat)

    d_loaders = get_dloaders(cfg, [vit_patch, vit_patch_idx, vggish, vggish_idx, clip_qst, clip_qst_idx, clip_word, word_mask, clip_word_idx, frame_feat, frame_feat_idx])
    model = get_model(cfg, device)
    
    logger.info(f"\n-------------- evaluating test dataset {cfg.data.test_annot} --------------")
    test(cfg, device, d_loaders['test'], model)
    

if __name__ == '__main__':
    main()