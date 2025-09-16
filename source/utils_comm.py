import random
import numpy as np
import torch

from source.args import args

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