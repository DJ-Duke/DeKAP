import math
import torch
import torch.nn as nn

from source.args import args

def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, args.mode)
    gain = nn.init.calculate_gain(args.nonlinearity)
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std


def kaiming_uniform(module):
    nn.init.kaiming_uniform_(
        module.weight, mode=args.mode, nonlinearity=args.nonlinearity
    )


def default(module):
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
