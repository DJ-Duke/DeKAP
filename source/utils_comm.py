import random
import numpy as np
import torch
import pathlib
from source.args import args

def add_changes_to_layer(module):
    if hasattr(module, 'add_changes'):
        module.add_changes()

def set_seeds(given_seed=None):
    if given_seed is None: 
        seed = args.seed
    else:
        seed = given_seed
    if getattr(args, 'fix_np_seed', False):
        np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def load_from_ckp(model, dir, model_name):
    resume_run_base_dir = pathlib.Path(f"{dir}/{model_name}.pt")
    checkpoint = torch.load(resume_run_base_dir, map_location=f"cuda:{args.multigpu[0]}")
    # 只保留当前模型中存在的参数
    pretrained_dict = {
        k: v for k, v in checkpoint.items() if k in model.state_dict()
    }
    model.load_state_dict(pretrained_dict)
    print(f"=> Loaded checkpoint '{resume_run_base_dir}'")

def naming_layers(model):
    module_index = 0
    for n, m in model.named_modules():
        if hasattr(m, "sparsity"):
            m.module_name = n
            m.module_index = module_index
            module_index += 1
            print(f"Layer {module_index}: {n}")

def scheduler_list_update(scheduler):
    if isinstance(scheduler, list):
        for s in scheduler:
            if s is not None:
                s.step()
    else:
        scheduler.step()