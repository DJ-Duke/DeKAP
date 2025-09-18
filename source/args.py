from copy import deepcopy
import argparse
import sys
import yaml
import os

args = None

def parse_arguments():
    parser = argparse.ArgumentParser(description="DeKAP")
    parser.add_argument(
        "--config", type=str, default=None, help="Config file to use, YAML format"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="Which optimizer to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        metavar="M",
        help="Weight decay (default: 0.0001)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=4, help="which gpu to use"
    )
    parser.add_argument("--name", type=str, default="default", help="Experiment id.")
    parser.add_argument(
        "--data", type=str, help="Location to store data",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Location to logs/checkpoints",
    )
    parser.add_argument("--resume", type=str, default=None, help='optionally resume')
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument(
        "--conv_type", type=str, default="StandardConv", help="Type of conv layer"
    )
    parser.add_argument(
        "--bn_type", type=str, default="StandardBN", help="Type of batch norm layer."
    )
    parser.add_argument(
        "--conv-init",
        type=str,
        default="default",
        help="How to initialize the conv weights.",
    )
    parser.add_argument("--model", type=str, help="Type of model.")
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument(
        "--trainer",
        default=None,
        type=str,
        help="Which trainer to use, default in trainers/default.py",
    )
    parser.add_argument(
        "--save", action="store_true", default=False, help="save checkpoints"
    )
    parser.add_argument("--no-scheduler", action="store_true", help="constant LR")
    parser.add_argument(
        "--config_name", default='', type=str,
    )
    parser.add_argument(
        "--config_root", default='', type=str,
    )
    parser.add_argument(
        "--wandb_name", default='', type=str,
    )
    parser.add_argument(
        "--flag_directly_load", type=bool, default=True,
    )
    args, _ = parser.parse_known_args()
    if args.config is not None:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    try:
        # load yaml file
        # 打印当前目录
        print(f"[!] Current directory: {os.getcwd()}")
        with open(args.config) as f:
            loaded_yaml = yaml.safe_load(f)
        
        # 然后用当前配置更新args
        args.__dict__.update(loaded_yaml)
        print(f"=> Reading YAML config from {args.config}")
            
        return args
            
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        raise

def run_args():
    global args 
    if args is None:
        args = parse_arguments()

def config_set_args(kwargs):
    global args
    args.__dict__['config'] = kwargs['config']
    get_config(args)

    for flag, val in kwargs.items():
        # 把flag中的-替换为_
        flag = flag.replace('-', '_')
        args.__dict__[flag] = val

run_args() 