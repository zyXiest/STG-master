from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import os
import glob
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import clip  
import argparse
import clip_net.clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_net.clip.load("ViT-L/14@336px", device=device)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
def interpolate_to_fixed_frames(img_list, fixed_frame_num=60):
    original_frame_num = len(img_list)
    if original_frame_num == 0:
        return []

    if original_frame_num == fixed_frame_num:
        return img_list

    target_indices = np.linspace(0, original_frame_num - 1, fixed_frame_num)
    interp_indices = np.round(target_indices).astype(int)


    interp_indices = np.clip(interp_indices, 0, original_frame_num - 1)

    return [img_list[i] for i in interp_indices]


def clip_feat_extract(img):

    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def ImageClIP_feat_extract(dir_fps_path, dst_clip_path):

    video_list = os.listdir(dir_fps_path)
    
    video_idx = 0
    total_nums = len(video_list)
    C = 768
    num_frames = 60

    for video in video_list:

        video_idx = video_idx + 1
        print("\n--> ", video_idx, video)

        save_file = os.path.join(dst_clip_path, video + '.npy')
        if os.path.exists(save_file):
            print(video + '.npy', "is already processed!")
            continue

        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))
        total_frames = len(video_img_list)
  
        sample_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        img_list = [video_img_list[i] for i in sample_indices]

        img_features = torch.zeros(len(img_list), C)

        idx = 0
        for img_cont in img_list:
            img_idx_feat = clip_feat_extract(img_cont)
            img_features[idx] = img_idx_feat
            idx += 1

        img_features = img_features.float().cpu().numpy()
        np.save(save_file, img_features)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx, " ----- save shape: ", img_features.shape)


if __name__ == "__main__":
    import sys
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    sys.path.append(ROOT.as_posix())

    import torch
    import src.tome as tome
    import timm

    dir_fps_path = '/home/WorkSpace/AVQA/Dataset/ActivityNet/Dataset/ActivityNet/ActNet-frames'
    dst_clip_path = "./feats/frame_feat"
    model_type = 'vitl14'


    ImageClIP_feat_extract(dir_fps_path, dst_clip_path)
