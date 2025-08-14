from pathlib import Path

import numpy as np
import torch.nn as nn
import os
import glob
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def get_tome_feat(
    model_size: str = 'base',
    patch_size: int = 32,
    image_size: int = 224,
    save_dir='./data/feats/visual_tome14',
    num_frames=60,
    transform: transforms.Compose = None,
):
    model = timm.create_model(
        f'vit_{model_size}_patch{patch_size}_{image_size}',
        pretrained=True
    ).to('cuda')
    model.head = Identity()
    model.global_pool = None
    tome.patch.timm(model, trace_source=True)
    model = model.to('cuda')
    
    if patch_size == 16 and image_size == 384:
        # model.r = [25] * 23
        model.r = [24] * 22
    elif patch_size == 32 and image_size == 224:
        model.r = 0

    transform = transform if transform is not None \
            else transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                         std=IMAGENET_DEFAULT_STD),
                ])

    
    # output shape check
    data = torch.randn(1, 3, image_size, image_size).to('cuda')
    output = model(data)
    print(output.shape)
    
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    dir_fps_path = '/home/WorkSpace/AVQA/Dataset/MUSIC-AVQA/frames-1ps/'
    video_list = os.listdir(dir_fps_path)
    video_idx = 0

    total_frames = 60
    frame_indices = np.linspace(0, total_frames-1, num_frames, total_frames, dtype=int).astype(int)

    for video in video_list:

        video_idx = video_idx + 1
        print("\n--> ", video_idx, video)

        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, '*.jpg')))

        video_img_list = [video_img_list[i] for i in frame_indices]

        save_path = f'{save_dir}/{video}.npy'
        if Path(save_path).exists():
            print(f"File {save_path} exists")
            continue

        data = torch.stack([
                transform(Image.open(video_img_list[i]).convert('RGB'))
                for i in range(len(video_img_list))
            ], dim=0).to('cuda').float(),
        
        data = data[0]
        B = 1
        T = num_frames
        with torch.no_grad():
            output = model(data.reshape(B*T, *data.shape[1:]))
            # output = output.reshape(B, T, *output.shape[1:])
        
        print(f"File {save_path} saved")
        np.save(save_path, output.cpu().numpy().astype(np.float16))
    


if __name__ == "__main__":
    import sys
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    sys.path.append(ROOT.as_posix())

    import torch
    import src.tome as tome
    import timm
    
    model_type = 'vitb32'
    if model_type == "vitb32":
        get_tome_feat(model_size='base', patch_size=32, image_size=224,
                        save_dir="./data/feats/Tome_b32")
        
    elif model_type == "vitl14":

        get_tome_feat(model_size='large', patch_size=16, image_size=384,
                        save_dir="./data/feats/Tome_l14")
