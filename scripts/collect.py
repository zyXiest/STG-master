import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import glob
import json
import clip
import h5py


def npy_to_hdf5(npy_dir, hdf5_path):

    video_idx = 0
    with h5py.File(hdf5_path, 'w') as hdf5_file:
 
        for filename in os.listdir(npy_dir):
            if filename.endswith('.npy'):
                video_idx += 1
                file_path = os.path.join(npy_dir, filename)

                data = np.load(file_path).astype(np.float16)

                dataset_name = os.path.splitext(filename)[0]

                hdf5_file.create_dataset(dataset_name, data=data)
                print("\n--> ", video_idx, dataset_name)

npy_dir = './data/feats/Tome_l14/'
hdf5_path = './data/feats/' +'tomebl14.h5'   
npy_to_hdf5(npy_dir, hdf5_path)
