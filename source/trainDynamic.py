import datetime
from datetime import datetime
import os
import pathlib
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, SequentialLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms
import wandb  
from args import args
import data
import schedulers
import trainers
from trainers.trainer_ps import test_anomaly_detection
import utils
import time

import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from source.utils_comm import *
from data.eval_datasets_core import DistillDataset, TransformDistillDataset
from z_paper_evaluations.eval_1 import get_distill_dataloader

# 设置工作目录为当前文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 打印当前运行目录
print(os.getcwd())

def set_args_all_case_false():
    args.flag_lora_proposed = False
    args.flag_using_equalizer = False
    args.flag_using_toplayer = False
    args.flag_using_parameter_alignment = False
    args.flag_using_data_adaptation = False
    args.flag_using_full_finetune = False
    args.flag_ordered_dropout = False
    args.add_str_to_case_dir = ""
    args.to_use_full_train_dataloader = False
    args.to_use_transformed_dataloader = False

def get_part_distill_dataloader(distill_dataloader, subset_indices, batch_size_factor=1):
    distill_dataset = distill_dataloader.dataset
    subset_dataset = torch.utils.data.Subset(distill_dataset, subset_indices)
    kwargs = {"num_workers": 0, "pin_memory": False, "shuffle": True, "batch_size": 1 * batch_size_factor}
    part_distill_dataloader = torch.utils.data.DataLoader(subset_dataset, **kwargs)
    return part_distill_dataloader

def transform_distill_dataloader(model, distill_dataloader, batch_size_factor=1):
    model.apply(lambda m: setattr(m, "pretrain_zfft", True))
    model.eval()
    data_list = []
    label_list = []
    pbar = tqdm(distill_dataloader, desc='Transforming Dataloader', ncols=120)
    for data, label in pbar:
        _, data_recon, _ = model(data)
        _, label_recon, _ = model(label)
        data_list.append(data_recon)
        label_list.append(label_recon)
    data_tensor = torch.cat(data_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)
    transform_distill_dataset = TransformDistillDataset(data_tensor, label_tensor)
    transform_distill_dataset.calculate_variance(1024)

    kwargs = {"num_workers": 0, "pin_memory": False, "shuffle": True, "batch_size": 1 * batch_size_factor}
    transform_distill_dataloader = torch.utils.data.DataLoader(transform_distill_dataset, **kwargs)
    model.apply(lambda m: setattr(m, "pretrain_zfft", False))
    return transform_distill_dataloader


def create_run_base_dir_task(task_name):
    run_base_dir = pathlib.Path(f"{args.log_dir}/{args.name}/{task_name}")
    os.makedirs(run_base_dir, exist_ok=True)
    return run_base_dir

def prepare_for_lora_training(model):
    set_model_sparsity_all_zero(model)
    model.apply(lambda m: change_with_lora_state_to_layer(m, True))
    # 进行压缩的过程。

    # 预先对model引入lora进行一些必要的设置。
    model.apply(backup_changes_to_layer)
    # model.apply(add_lora_changes_to_layer)
    model.apply(obtain_full_ft_sigvalue_to_layer)
    model.module.set_no_change_vq(args.no_change_vq)
    model.module.set_finetune_vq(args.finetune_vq)


def prepare_criterion():
    criterion = F.mse_loss
    # 设置perception的优化criterion
    import torchvision.models as models
    from torchvision.models import VGG16_Weights
    # 使用新的weights参数替代已废弃的pretrained参数
    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    IFL_feature_extractor = nn.Sequential(*list(vgg.children())[:16]).to(args.device)
    IFL_feature_extractor.eval()
    return criterion, IFL_feature_extractor


def multi_level_DK_distill(task_name, flag_directly_load=True):
    '''
    task_name: 具体任务的名称
    flag_directly_load： 直接加载经过预处理的数据集
    #TODO 2025-09-16 工作到这里
    '''
    print(f"[INFO] parameter compression level: {args.loraSra_paraBgt_list}, learning rate: {args.lr}")
    run_base_dir = create_run_base_dir_task(task_name)
    args.run_base_dir = run_base_dir
    print(f"[INFO] SAVE DIR: {run_base_dir}")
    set_seeds()
    
    wandb.init(
        project="DeKAP-rev1-trainDynamic",
        name=args.wandb_name+f"{task_name}_{datetime.now().strftime('%m%d_%H%M')}",
        config=locals(),
        notes=""
    )

    # 获取模型
    model = utils.get_model()
    model = utils.set_gpu(model)
    model.apply(add_changes_to_layer)
    # 加载模型
    preft_model_dir = getattr(args, f"pre_ft_dir_case_{task_name}")
    load_from_ckp(model, dir=preft_model_dir, model_name=args.pre_ft_ckp_name)
    # 模型基础设置，保证运行在FT的阶段。
    model.apply(lambda m: setattr(m, "task", 0)) # 使用第几个stored change来进行训练
    model.apply(lambda m: setattr(m, "pretrain_zfft", False)) # 告诉模型当前是finetune段。
    naming_layers(model)

    set_model_sparsity_all_zero(model, all_ones=True)
    model.apply(lambda m: change_with_lora_state_to_layer(m, False))

    # 使用数据集, 这里假设通过eval_1已经生成了distill dataset，所以直接加载。
    #ANCHOR 1 获取数据集的部分
    directly_load = flag_directly_load
    if flag_directly_load is False:
        data_loader = getattr(data, args.set)()
        distill_dataloader_train = get_distill_dataloader(task_name, model, data_loader.train_loader, "train", directly_load, flag_use_ori_label)
        distill_dataloader_test = get_distill_dataloader(task_name, model, data_loader.val_loader, "test", directly_load, flag_use_ori_label)
    else:
        distill_dataloader_train = get_distill_dataloader(task_name, model, None, "train", directly_load, flag_use_ori_label)
        distill_dataloader_test = get_distill_dataloader(task_name, model, None, "test", directly_load, flag_use_ori_label)

    trainer = getattr(trainers, args.trainer)
    train, test = trainer.train, trainer.test
    criterion, IFL_feature_extractor = prepare_criterion()

    # 测试一下，如果没有和任务对齐的话，也就是使用原始预训练参数的话，loss是怎么样的。
    model.apply(lambda m: setattr(m, "pretrain_zfft", True)) 
    test_total_loss = test(model, writer, criterion, distill_dataloader_test, 0, verbose=True, flag_draw_example_images=False, criterion_IFL=IFL_feature_extractor)
    writer.add_scalar("Eval2b/Test_Epoch_Loss", test_total_loss, 0)
    wandb.log({f"test/Test_Epoch_Loss": test_total_loss}, step=0)
    model.apply(lambda m: setattr(m, "pretrain_zfft", False))

    prepare_for_lora_training(model)
    paraBgt_list = args.loraSra_paraBgt_list
    prev_rank_list = None
    sigvalueVparam_mat, cumsum_parameter_ratio_vec, module_list, total_num_params = get_sigvalueVparam_mat(model)

    train_epochs = args.epochs_loraDstill
    cur_best_epoch = 0
    total_epochs = 0

    # 进行lora的训练过程
    assert len(paraBgt_list) == 1, "在trainDynamic中，只使用一个rank plan。从而方便对比出来影响"
    last_bgt = paraBgt_list[-1]
    _, max_rank_list = allocate_rank_given_mat_and_index(sigvalueVparam_mat, cumsum_parameter_ratio_vec, last_bgt, module_list, total_num_params, prev_rank_list)
    for idx, m in enumerate(module_list):
        adding_rank = max_rank_list[idx]
        # adding rank to layer
        m.add_new_rank_list(adding_rank)

    plan_list = [[] for _ in range(len(module_list))]
    complete_rank_plan_list = []
    for cur_bgt in paraBgt_list:
        new_rank_list, _ = allocate_rank_given_mat_and_index(sigvalueVparam_mat, cumsum_parameter_ratio_vec, cur_bgt, module_list, total_num_params, prev_rank_list, writer=writer)
        complete_rank_plan_list.append(new_rank_list)
        prev_rank_list = new_rank_list
        for idx, m in enumerate(module_list):
            plan_list[idx].append(new_rank_list[idx])
    
    for idx, m in enumerate(module_list):
        m.set_target_rank_plan(plan_list[idx])

    total_epochs = 0
    optimizer, scheduler = gen_optimizer_and_scheduler_loraSRA_list(model, module_list, new_rank_list, None, using_one_optimizer_shcduler=True)
    cur_min_loss = 1000.0
    for epoch in range(1, train_epochs * len(paraBgt_list) + 1):
        # 训练
        total_epochs += 1
        train(model, writer, distill_dataloader_train, optimizer, criterion, total_epochs, None, verbose=False, random_rank_plans=len(paraBgt_list), criterion_IFL=IFL_feature_extractor)

        # 这个评估将会非常耗时。
        rkp = np.mod(total_epochs, len(paraBgt_list))
        test_total_loss = test(model, writer, criterion, distill_dataloader_test, total_epochs, verbose=True, flag_draw_example_images=False, rank_plan=rkp, criterion_IFL=IFL_feature_extractor)
        wandb.log({f"test/Test_Epoch_Loss": test_total_loss}, step=total_epochs)
        
        if test_total_loss < cur_min_loss:
            cur_min_loss = test_total_loss
            print(f"!!! Current best PT at Epoch {total_epochs}, Test Total Loss: {test_total_loss:.2f}")
            writer.add_text("FT_curBest", f"{test_total_loss:.2f} at total_epochs {total_epochs}", total_epochs)
            if args.save_curbest_model:
                torch.save(model.state_dict(), run_base_dir / f"LoRA_curBest.pt")
            cur_best_epoch = total_epochs
        scheduler_list_update(scheduler)
        writer.add_scalar("Test/CurBestLoss", cur_best_epoch, total_epochs)
        wandb.log({f"test/CurBestLoss": cur_best_epoch}, step=total_epochs)
        writer.flush()
    wandb.finish()
    return
