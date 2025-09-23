#!/bin/bash
# bash run_script/debug.sh
# Activate the virtual environment
source .venv/bin/activate

# python source/trainDynamic.py --task_list resEnhance --flag_directly_load True
python process/distillation.py \
    --gpu_id 0 \
    --wandb_name distill_test \
    --loraSra_paraBgt_list="[1,3,5]" \
    --lr 0.0005 \
    --flag_directly_load True
