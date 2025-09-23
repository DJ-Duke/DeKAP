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
from tqdm import tqdm

from pathlib import Path

# 获取当前文件的上级目录
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.append(str(PROJECT_ROOT))
print(f"[!] Project root is now set to: {PROJECT_ROOT}")

from source.trainDynamic import multi_level_DK_distill
from source.args import args, config_set_args

def main():
    fire.Fire(main_program)

def main_program(gpu_id: int, # the index of the GPU to use
                 wandb_name: str="distill_test", # wandb name
                 loraSra_paraBgt_list: list[int]=[1], # loraSra_paraBgt_list is the compression levels, 1 means the distilled knowledge is 1% of the total parameters
                 lr: float=0.0005, # learning rate
                 flag_directly_load: bool=True, # directly load the preprocessed dataset
                 config_name: str="base_config", # the name of theconfig file to use
                 ):
    print(f"at main_program, loraSra_paraBgt_list: {loraSra_paraBgt_list}")
    GPU_ID = gpu_id
    print(f"[!] Using GPU {GPU_ID}")
    args.config_root = "./configs/"
    args.config_name = config_name
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
        args.add_str_to_run_base_dir = f"_{datetime.now().strftime('%m%d_%H%M')}"
        multi_level_DK_distill(task, flag_directly_load)

if __name__ == "__main__":
    main()