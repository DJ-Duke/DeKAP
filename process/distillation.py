import datetime
from datetime import datetime
from copy import deepcopy
from multiprocessing import Process, Queue
from itertools import product
import sys, os
import numpy as np
import time
import argparse
import torch
import pathlib
import fire

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from source.trainDynamic import multi_level_DK_distill
from source.args import args, config_set_args

def main():
    fire.Fire(main_program)

def main_program(gpu_id: int, # 使用的GPU的序号
                 wandb_name: str="distill_test", # wandb name
                 loraSra_paraBgt_list: list[int]=[1], # loraSra_paraBgt_list 是 compression levels, 1代表 distilled knowledge 参数量为原总参数的1%
                 lr: float=0.0005, # learning rate
                 flag_directly_load: bool=True, # 直接加载经过预处理的数据集
                 ):
    GPU_ID = gpu_id
    print(f"[!] Using GPU {GPU_ID}")
    args.config_root = "./configs/"
    config_name = args.config_name
    config_path = args.config_root + config_name + ".yaml"
    print(f"[!] Using config {config_path}")
    args.wandb_name = wandb_name
    args.loraSra_paraBgt_list = loraSra_paraBgt_list

    config_set_args({"config": config_path, "multigpu": [GPU_ID]})
    eval_name = f"{config_name}~sd{args.seed}"
    args.name = eval_name

    task_to_nameTransfer_dict = {
        "resEnhance": "low_resolution",
        "colorCorrect": "color",
        "noiseRemove": "noise",
        "anomalyDetect": "anomaly",
    }

    task_list = args.task_list
    
    for task in task_list:
        print(f"[INFO] Current task: {task}")
        args.name_transform = task_to_nameTransfer_dict[task]
        args.loraSra_paraBgt_list = loraSra_paraBgt_list
        args.lr = lr
        args.flag_directly_load = flag_directly_load
        # main_eval_2_rev1(task,flag_directly_load)
        args.add_str_to_run_base_dir = f"_{datetime.now().strftime('%m%d_%H%M')}"
        multi_level_DK_distill(task, flag_directly_load)

