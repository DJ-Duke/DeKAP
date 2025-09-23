#!/bin/bash
# bash run_script/run_distillation.sh

source .venv/bin/activate

python process/distillation.py \
    --gpu_id 0 \
    --wandb_name distill_test \
    --loraSra_paraBgt_list="[1,3,5]" \
    --lr 0.0005 \
    --flag_directly_load True
