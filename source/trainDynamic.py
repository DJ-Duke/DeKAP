import datetime
from datetime import datetime
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb  
import fire

from .args import args
import trainers
from .utils import *
from .utils_comm import *

def create_run_base_dir_task(task_name):
    run_base_dir = pathlib.Path(f"{args.log_dir}/{args.name}/{task_name}")
    os.makedirs(run_base_dir, exist_ok=True)
    return run_base_dir

def prepare_for_lora_training(model):
    # the preparation process of compression.
    model.apply(lambda m: hasattr(m, 'with_lora_change') and setattr(m, 'with_lora_change', True))
    # the necessary settings for the model to introduce lora.
    model.apply(lambda m: m.backup_changes() if hasattr(m, 'backup_changes') else None)
    
    # model.apply(add_lora_changes_to_layer) 
    model.apply(lambda m: m.obtain_full_ft_sigvalue() if hasattr(m, 'obtain_full_ft_sigvalue') else None)
    model.module.set_no_change_vq(args.no_change_vq)
    model.module.set_finetune_vq(args.finetune_vq)


def prepare_criterion():
    criterion = F.mse_loss
    # set the optimization criterion for perception
    import torchvision.models as models
    from torchvision.models import VGG16_Weights
    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    IFL_feature_extractor = nn.Sequential(*list(vgg.children())[:16]).to(args.device)
    IFL_feature_extractor.eval()
    return criterion, IFL_feature_extractor


def multi_level_DK_distill(task_name, flag_directly_load=True, flag_use_ori_label=False):
    '''
    task_name: the name of the specific task
    flag_directly_load: directly load the preprocessed dataset
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

    # get the model
    model = get_model()
    model = set_gpu(model)
    model.apply(add_changes_to_layer)
    # load the model
    preft_model_dir = getattr(args, f"pre_ft_dir_case_{task_name}")
    load_from_ckp(model, dir=preft_model_dir, model_name=args.pre_ft_ckp_name)
    # model basic settings, ensure the model is running in the FT stage.
    model.apply(lambda m: setattr(m, "change_idx", 0)) # use the stored chagne of the given index for training
    model.apply(lambda m: setattr(m, "pretrain", False)) # tell the model that current is the finetune stage, not the pretrain stage.
    naming_layers(model) # 命名模型中的各层并标记序号。

    model.apply(lambda m: hasattr(m, 'with_lora_change') and setattr(m, 'with_lora_change', True))

    # use the dataset, here it is assumed that the distill dataset has been generated, so it can be loaded directly.
    assert flag_directly_load is True, "in trainDynamic, only direct loading of the dataset is allowed."
    #get the dataset
    directly_load = flag_directly_load
    if flag_directly_load is False:
        raise NotImplementedError("[ERROR] only the preprocessed dataset can be loaded directly, future code release will support this.")
    else:
        distill_dataloader_train = get_distill_dataloader(task_name, model, None, "train", directly_load, flag_use_ori_label)
        distill_dataloader_test, distill_dataloader_draw = get_distill_dataloader(task_name, model, None, "test", directly_load, flag_use_ori_label)

    trainer = getattr(trainers, args.trainer)
    train, test, drawer = trainer.train, trainer.test, trainer.show_comparison_results
    criterion, IFL_feature_extractor = prepare_criterion()

    # test the non-alignment loss when using the original pretrained parameters.
    model.apply(lambda m: setattr(m, "pretrain", True)) 
    test_total_loss = test(model, criterion, distill_dataloader_test, 0, verbose=True, criterion_IFL=IFL_feature_extractor)
    drawer(model, distill_dataloader_draw, epoch=0, rank_plan=None)
    model.apply(lambda m: setattr(m, "pretrain", False))

    prepare_for_lora_training(model)
    paraBgt_list = args.loraSra_paraBgt_list
    prev_rank_list = None
    sigvalueVparam_mat, cumsum_parameter_ratio_vec, module_list, total_num_params = get_sigvalueVparam_mat(model)

    train_epochs = args.epochs_loraDstill
    cur_best_epoch = 0
    total_epochs = 0

    # the lora training process
    last_bgt = paraBgt_list[-1]
    _, max_rank_list = allocate_rank_given_mat_and_index(sigvalueVparam_mat, cumsum_parameter_ratio_vec, last_bgt, module_list, total_num_params, prev_rank_list)
    for idx, m in enumerate(module_list):
        adding_rank = max_rank_list[idx]
        # adding rank to layer
        m.add_new_rank_list(adding_rank)

    plan_list = [[] for _ in range(len(module_list))]
    complete_rank_plan_list = []
    for cur_bgt in paraBgt_list:
        new_rank_list, _ = allocate_rank_given_mat_and_index(sigvalueVparam_mat, cumsum_parameter_ratio_vec, cur_bgt, module_list, total_num_params, prev_rank_list)
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
        # training
        total_epochs += 1
        train(model, distill_dataloader_train, optimizer, criterion, total_epochs, verbose=False, random_rank_plans=len(paraBgt_list), criterion_IFL=IFL_feature_extractor)

        # rkp means "rank plan"
        rkp = np.mod(total_epochs, len(paraBgt_list))
        test_total_loss = test(model, criterion, distill_dataloader_test, total_epochs, verbose=True, rank_plan=rkp, criterion_IFL=IFL_feature_extractor)
        if epoch % 10 == 0:
            for rkp in range(len(paraBgt_list)):
                print(f"!!! Draw example images at epoch {total_epochs}, rank plan {rkp}")
                drawer(model, distill_dataloader_draw, epoch=total_epochs, rank_plan=rkp)
        # update the scheduler
        scheduler_list_update(scheduler)
        wandb.log({f"test/CurBestEpoch": cur_best_epoch}, step=total_epochs)
    wandb.finish()
    return

def main():
    fire.Fire(multi_level_DK_distill)

if __name__ == "__main__":
    main()