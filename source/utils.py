from PIL import Image
import time
import pathlib

import torch
import torch.nn as nn
import models
import models.module_util as module_util
import torch.backends.cudnn as cudnn

from args import args
import numpy as np


def calculate_psnr(data, data_recon):
    with torch.no_grad():
        data_numpy = data.detach().cpu().float().numpy()
        data_numpy = (np.transpose(data_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        data_int8 = data_numpy.astype(np.uint8)
        
        recon_numpy = data_recon.detach().cpu().float().numpy() 
        recon_numpy = (np.transpose(recon_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        recon_int8 = recon_numpy.astype(np.uint8)
        
        diff = np.mean((np.float64(recon_int8) - np.float64(data_int8))**2, (1, 2, 3))
        batch_psnr = 10 * np.log10(255.0**2 / np.mean(diff))
        return batch_psnr

def freeze_model_weights(model: nn.Module):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Freezing weight for {n}")
            m.weight.requires_grad_(False)

            if m.weight.grad is not None:
                m.weight.grad = None
                print(f"==> Resetting grad value for {n} -> None")


def freeze_model_scores(model: nn.Module, task_idx: int):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Freezing weight for {n}")
            m.scores[task_idx].requires_grad_(False)

            if m.scores[task_idx].grad is not None:
                m.scores[task_idx].grad = None
                print(f"==> Resetting grad value for {n} scores -> None")


def unfreeze_model_weights(model: nn.Module):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Unfreezing weight for {n}")
            m.weight.requires_grad_(True)


def unfreeze_model_scores(model: nn.Module, task_idx: int):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"=> Unfreezing weight for {n}")
            m.scores[task_idx].requires_grad_(True)


def set_gpu(model):
    if args.multigpu is None:
        args.device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )
        args.device = torch.cuda.current_device()
        cudnn.benchmark = True

    return model


def get_model():
    model = models.__dict__[args.model]()
    return model





