import glob
import numpy as np
import os
import torch
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
import torch.utils.data as data
import copy
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from source.args import args

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Borrowed from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
class DistillDataset(data.Dataset):
    def __init__(self, data_np, data_recon_np):
        # 修改1：确保数据类型一致且为float32
        self.data_np = torch.from_numpy(data_np).float()
        self.data_recon_np = torch.from_numpy(data_recon_np).float()
        
        # 修改2：如果有GPU，预先将数据移到GPU
        # if torch.cuda.is_available():
        self.data_np = self.data_np.to(args.device)
        self.data_recon_np = self.data_recon_np.to(args.device)
        # 修改3：预计算方差
        self._variance = None

    def __getitem__(self, index):
        # 修改4：直接返回tensor，避免每次创建新的numpy数组
        return self.data_np[index], self.data_recon_np[index]

    def __len__(self):
        return self.data_np.shape[0]
    
    def calculate_variance(self, batch_size):
        # 修改5：使用torch操作替代numpy，避免CPU-GPU数据传输
        if self._variance is None:
            with torch.no_grad():
                random_indices = torch.randint(0, len(self.data_recon_np), (batch_size,))
                if torch.cuda.is_available():
                    random_indices = random_indices.cuda()
                image_batch = self.data_recon_np[random_indices]
                self._variance = torch.var(image_batch).item()
        return self._variance


