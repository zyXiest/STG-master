import sys
sys.path.append('..')
sys.path.append('.')
import argparse
from config import Constants
import os
import pickle
from misc import utils_corpora

import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import glob
import json
import clip
import h5py

import os
import subprocess
from tqdm import tqdm
import pickle
import shutil
import argparse
import glob
import json
import os
import cv2
import numpy as np
from glob import glob
import subprocess
from scipy.interpolate import interp1d

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_a', type=str, default='dataset/val_a.json',
                        help='path to the json file containing your prediction')
    parser.add_argument('--train_a', type=str, default='dataset/train_a.json',
                        help='path to the json file containing the ground true')
    parser.add_argument('--test_a', type=str, default='dataset/test_a.json',
                        help='path to the json file containing the ground true')
    params = parser.parse_args()
    return params



if __name__ == "__main__":
    
    args = parse_opt()

    train_a = json.load(open(args.train_a, 'r'))
    val_a = json.load(open(args.val_a, 'r'))
    test_a = json.load(open(args.test_a, 'r'))

    answer = train_a #+ val_a + test_a
    all_answers = set()
    for item in answer:
        ans = item['answer']
        all_answers.add(ans)

    all_types = set()
    for item in answer:
        ans = item['type']
        all_types.add(ans)


    answer_list = sorted(list(all_answers))
    all_types = sorted(list(all_types))

    ans2id = {ans: idx for idx, ans in enumerate(answer_list)}
    id2ans = {idx: ans for ans, idx in ans2id.items()}


    vocab = {
        "ans2id": ans2id,
        "id2ans": id2ans
    }

    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


