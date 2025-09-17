from PIL import Image
import wandb
import time
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import models
import models.module_util as module_util
import torch.backends.cudnn as cudnn
import os
from args import args
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from data import DistillDataset
from source.utils_comm import *


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

def get_distill_dataloader(task_name, model, data_loader, str_train_test, directly_load=False, flag_use_ori_label=False):
    set_seeds(args.distill_dataset_seed) 
    save_path = f"{args.saveDir_distill_dataset}"
    os.makedirs(save_path, exist_ok=True)
    save_path = save_path + f"/{task_name}_{str_train_test}.npz"
    if directly_load:
        print(f"[INFO] Loading distill dataset from {save_path}")
        load_data = np.load(save_path, mmap_mode='r')
        data_np = load_data['data']
        data_recon_np = load_data['data_recon']
        if flag_use_ori_label:
            label_np = load_data['label']
        else:
            label_np = None
    else:
        raise NotImplementedError("[ERROR] 暂时只能使用已经打包好的数据集，后续代码正在整理中。")
    
    if flag_use_ori_label is False:
        distill_dataset = DistillDataset(data_np, data_recon_np)
    else:
        distill_dataset = DistillDataset(data_np, label_np)
    
    if str_train_test == "train":
        args.data_variance = distill_dataset.calculate_variance(1024)
    
    shuffle_flag = True if str_train_test == "train" else False
    batch_size = args.batch_size if str_train_test == "train" else args.test_batch_size
    kwargs = {"num_workers": 0, "pin_memory": False, "shuffle": shuffle_flag, "batch_size": batch_size}
    distill_dataloader = torch.utils.data.DataLoader(distill_dataset, **kwargs)
    if str_train_test == "test" and args.eval_index == 1:
        batch_size = 1 if getattr(args, "flag_output_test_set_image", False) else batch_size
        subset_indices = args.to_save_list
        if not subset_indices == 'None':
            subset_dataset = torch.utils.data.Subset(distill_dataset, subset_indices)
            draw_distill_dataloader = torch.utils.data.DataLoader(subset_dataset, **kwargs)
        else:
            draw_distill_dataloader = torch.utils.data.DataLoader(distill_dataset, **kwargs)
        return distill_dataloader, draw_distill_dataloader
    else:
        return distill_dataloader

def get_sigvalueVparam_mat(model):
    # 是 allocate_model_lora_rank 函数的一部分，只获取 那个用来进行lora rank分配的参考矩阵，而暂时不实质上进行分配。
    total_num_params = 0
    crosslayer_sigvalue_NparaRank_mat = np.zeros((0,3))
    module_list = []
    module_index = 0
    
    for n, m in model.named_modules():
        if hasattr(m, "parameter_per_rank"):
            m.module_name = n
            module_list.append(m)

            total_num_params += m.weight.numel()
            
            # 构建每层的矩阵 [sigvalue, param_per_rank, module_index]
            num_ranks = len(m.fullft_sigvalue_vec)
            layer_mat = np.zeros((num_ranks, 3))
            layer_mat[:,0] = m.fullft_sigvalue_vec
            layer_mat[:,1] = m.parameter_per_rank
            layer_mat[:,2] = module_index
            
            crosslayer_sigvalue_NparaRank_mat = np.vstack((
                crosslayer_sigvalue_NparaRank_mat, 
                layer_mat
            ))
            module_index += 1
    
    for m_i, m in enumerate(module_list):
        m.module_index = m_i
    
    print(f"Total number of parameters: {total_num_params}, number of layers: {module_index}")
    
    # 按照奇异值（第一列）大小排序
    idx_sort = np.argsort(crosslayer_sigvalue_NparaRank_mat[:,0])[::-1]  # 降序排列
    sigvalueVparam_mat = crosslayer_sigvalue_NparaRank_mat[idx_sort]
    cumsum_parameter_ratio_vec = np.cumsum(sigvalueVparam_mat[:,1]) / total_num_params
    return sigvalueVparam_mat, cumsum_parameter_ratio_vec, module_list, total_num_params

def allocate_rank_given_mat_and_index(sigvalueVparam_mat, cumsum_parameter_ratio_vec, parameter_budget_, module_list, total_num_params, last_rank_list=None, writer=None):
    parameter_budget = parameter_budget_ / 100.
    idx_select = np.where(cumsum_parameter_ratio_vec > parameter_budget)[0][0]
    index_select_vec = sigvalueVparam_mat[:idx_select,2]
    
    new_rank_list = []
    rank_change_list = []

    if last_rank_list is None:
        last_rank_list = [0] * len(module_list)

    selected_param_num = 0
    sum_allocated_rank = 0
    for idx, m_i in enumerate(module_list):
        new_lora_rank_specific = np.sum(index_select_vec == idx)
        last_rank_specific = last_rank_list[idx]
        sum_allocated_rank += new_lora_rank_specific
        max_rank = m_i.maximum_rank
        if new_lora_rank_specific == max_rank:
            add_str = "full-rank"
        else:
            add_str = ""

        adding_param_num = np.sum(m_i.parameter_per_rank[:new_lora_rank_specific])
        selected_param_num += adding_param_num
        print(f"Allocate lora rank to layer {idx} {m_i.module_name} {last_rank_specific}->{new_lora_rank_specific}|{max_rank} {add_str}")

        new_rank_list.append(new_lora_rank_specific)
        rank_increase = new_lora_rank_specific - last_rank_specific
        rank_change_list.append(rank_increase)

    real_budget = selected_param_num / total_num_params
    print(f"====> INFO Selected parameter number: {selected_param_num}, parameter budget satisfies: {real_budget:.6f}% target budget: {parameter_budget:.6f}")
    wandb.log({"rank_list": str(new_rank_list)})
    return new_rank_list, rank_change_list

def gen_optimizer_and_scheduler_loraSRA_list(
    model, module_list, new_rank_list, prev_rank_list, using_one_optimizer_shcduler=False, add_last_layer_baseline=False):
    params_opt = []

    for n, p in model.named_parameters():
        p.requires_grad = False
    
    if prev_rank_list is None:
        prev_rank_list = [0] * len(module_list)

    for m_idx, m in enumerate(module_list):
        if hasattr(m, "parameter_per_rank"):
            starting_rank = prev_rank_list[m_idx]
            ending_rank = new_rank_list[m_idx]
            for i in range(starting_rank, ending_rank):
                m.lora_A_list[i].requires_grad = True
                m.lora_B_list[i].requires_grad = True
                params_opt.append(m.lora_A_list[i])
                params_opt.append(m.lora_B_list[i])
    
    if add_last_layer_baseline:
        params_opt.append(module_list[-1].changes[0])

    if using_one_optimizer_shcduler:
        milestone = [len(args.loraSra_paraBgt_list) * args.milestones_lora[0]]
    else:
        milestone = args.milestones_lora
    
    lr = args.lr
    optimizer_opt = optim.Adam(params_opt, lr=lr, weight_decay=args.wd)
    scheduler_opt = MultiStepLR(optimizer_opt, milestones=milestone, gamma=args.gamma_lora)
    
    return optimizer_opt, scheduler_opt