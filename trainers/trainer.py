import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import wandb

from source.args import args
from source.utils import calculate_psnr



def set_rank_plan_to_layer(module, rank_plan_idx):
    if hasattr(module, 'set_rank_plan'):
        module.set_rank_plan(rank_plan_idx)


def calcualte_IFL_loss(criterion_IFL, data_recon, data_target):
    real_feature = criterion_IFL(data_target)
    recon_feature = criterion_IFL(data_recon)
    loss = F.mse_loss(real_feature, recon_feature)
    return loss

def train(model, train_loader, optimizer, criterion, epoch, verbose=True, stop_iterations = -1, random_rank_plans=None, always_use_this_rank=None, criterion_IFL=None):
    model.zero_grad()
    model.train()

    num_batches = len(train_loader)
    train_loss = 0
    train_perplexity = 0
    train_psnr = 0
    train_recon_loss = 0
    num_samples = 0
    if criterion_IFL is not None:
        train_IFL_loss = 0

    if criterion == F.mse_loss:
        variance_coeff = args.data_variance
    elif criterion == F.l1_loss:
        variance_coeff = np.sqrt(args.data_variance)
    
    if criterion_IFL is not None:
        transform_for_IFL = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    pbar = tqdm(train_loader, desc='Training', ncols=120)
    for batch_idx, (data, label) in enumerate(pbar):

        if stop_iterations > 0 and batch_idx >= stop_iterations:
            break
        if args.iter_lim < 0 or batch_idx < args.iter_lim:
            if data.device != args.device:
                data = data.to(args.device)
                label = label.to(args.device)
            # target = data.clone()
            
            if isinstance(optimizer, list):
                for i in range(len(optimizer)):
                    if optimizer[i] is not None:
                        optimizer[i].zero_grad()
            else:
                optimizer.zero_grad()
            
            if random_rank_plans != None:
                if always_use_this_rank is not None:
                    random_plan_idx = always_use_this_rank
                else: # always_use_this_rank is None
                    random_plan_idx = np.random.randint(0, random_rank_plans)
                model.apply(lambda m: set_rank_plan_to_layer(m, random_plan_idx))
                

            if model.module.name == "VQVAE_ps":
                vq_loss, data_recon, perplexity = model(data)
                recon_error = criterion(data_recon, label) / variance_coeff
                loss = vq_loss + recon_error
                if criterion_IFL is not None:
                    if args.warmup_IFL:
                        current_lambda = min(args.lambda_IFL * (epoch - 1) / 10, args.lambda_IFL)
                    else:
                        current_lambda = args.lambda_IFL
                    data_recon_IFL = transform_for_IFL(data_recon)
                    label_IFL = transform_for_IFL(label)
                    IFL_loss = calcualte_IFL_loss(criterion_IFL, data_recon_IFL, label_IFL)
                    loss += current_lambda * IFL_loss
                loss_print = loss.item()
                add_str = "VQVAE"
                batch_psnr = calculate_psnr(label, data_recon)

                train_loss += loss.item() * len(data)
                train_perplexity += perplexity.item() * len(data)
                train_psnr += batch_psnr * len(data)
                train_recon_loss += recon_error.item() * len(data)
                if criterion_IFL is not None:
                    train_IFL_loss += IFL_loss.item() * len(data)
                num_samples += len(data)
            else:
                raise NotImplementedError(f"Model {model.module.name} not implemented")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if isinstance(optimizer, list):
                for i in range(len(optimizer)):
                    if optimizer[i] is not None:
                        optimizer[i].step()
            else:
                optimizer.step()

            pbar_string = f'Loss: {loss_print:.3f} ({recon_error.item():.3f}/{vq_loss.item():.3f}) Prplx: {perplexity:.3f} PSNR: {batch_psnr:.3f}dB'
            if random_rank_plans != None:
                pbar_string += f' RP: {random_plan_idx}'
            if criterion_IFL is not None:
                pbar_string += f' IFL: {IFL_loss.item():.3f} ({IFL_loss.item()*current_lambda:.3f})'
            pbar.set_postfix_str(
                pbar_string
            )
        else:
            break

    
    train_loss /= num_samples
    train_recon_loss /= num_samples
    train_perplexity /= num_samples
    train_psnr /= num_samples
    if criterion_IFL is not None:
        train_IFL_loss /= num_samples
    pbar.close()
    wandb.log({f"train/Total_loss": train_loss}, step=epoch)
    wandb.log({f"train/Recon_loss": train_recon_loss}, step=epoch)
    wandb.log({f"train/Perplexity": train_perplexity}, step=epoch)
    wandb.log({f"train/PSNR": train_psnr}, step=epoch)
    if criterion_IFL is not None:
        wandb.log({f"train/IFL_loss": train_IFL_loss}, step=epoch)
    print(f"=> Train Epoch {epoch}: Recon loss: {train_recon_loss:.4f}, Perplexity: {train_perplexity:.4f}, PSNR: {train_psnr:.4f}")

def test(model, criterion, test_loader, epoch, verbose=True, rank_plan=None, criterion_IFL=None):
    model.zero_grad()
    model.eval()
    test_loss = 0
    test_recon = 0
    test_perplexity = 0
    test_psnr = 0
    if criterion_IFL is not None:
        test_IFL_loss = 0
    num_batches = 0
    num_samples = 0
    flag_first_batch = True

    if criterion == F.mse_loss:
        variance_coeff = args.data_variance
    elif criterion == F.l1_loss:
        variance_coeff = np.sqrt(args.data_variance)
    
    if criterion_IFL is not None:
        transform_for_IFL = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    pbar = tqdm(test_loader, desc='Testing', ncols=120)
    if rank_plan != None:
        model.apply(lambda m: set_rank_plan_to_layer(m, rank_plan))
    with torch.no_grad():
        for data, label in pbar:
            if data.device != args.device:
                data = data.to(args.device)
                label = label.to(args.device)

            if model.module.name == "VQVAE_ps":
                vq_loss, data_recon, perplexity = model(data)
                batch_psnr = calculate_psnr(label, data_recon)
                recon_error = criterion(data_recon, label) / variance_coeff
                test_recon += recon_error.item() * len(data)
                test_loss += recon_error.item() * len(data)
                if criterion_IFL is not None:
                    data_recon_IFL = transform_for_IFL(data_recon)
                    label_IFL = transform_for_IFL(label)
                    IFL_loss = calcualte_IFL_loss(criterion_IFL, data_recon_IFL, label_IFL)
                    current_lambda = args.lambda_IFL
                    test_IFL_loss += IFL_loss.item() * len(data) * current_lambda
                    test_loss += IFL_loss.item() * len(data) * current_lambda
                test_perplexity += perplexity.item() * len(data)
                add_str = "VQVAE"
            else:
                raise NotImplementedError(f"Model {model.module.name} not implemented")
        

            test_psnr += batch_psnr * len(data)
            if criterion_IFL is not None:
                test_IFL_loss += IFL_loss.item() * len(data)
            num_batches += 1
            num_samples += len(data)

            pbar_string = f'Loss: {recon_error.item():.3f} Vq loss: {vq_loss.item():.3f} Perplexity: {perplexity:.3f} PSNR: {batch_psnr:.3f}dB'
            if criterion_IFL is not None:
                pbar_string += f' IFL: {IFL_loss.item():.3f}'
            if rank_plan != None:
                pbar_string += f' RP: {rank_plan}'
            pbar.set_postfix_str(
                pbar_string
            )

    pbar.close()
    test_loss /= num_samples
    test_recon /= num_samples
    test_perplexity /= num_samples
    test_psnr /= num_samples
    if criterion_IFL is not None:
        test_IFL_loss /= num_samples
    if rank_plan is not None:
        wandb.log({f"test/Total_loss/{rank_plan}": test_loss}, step=epoch)
        wandb.log({f"test/Recon_loss/{rank_plan}": test_recon}, step=epoch)
        wandb.log({f"test/Perplexity/{rank_plan}": test_perplexity}, step=epoch) 
        wandb.log({f"test/PSNR/{rank_plan}": test_psnr}, step=epoch)
        if criterion_IFL is not None:
            wandb.log({f"test/IFL_loss/{rank_plan}": test_IFL_loss}, step=epoch)
    else:
        wandb.log({"test/Total_loss": test_loss}, step=epoch)
        wandb.log({"test/Recon_loss": test_recon}, step=epoch)
        wandb.log({"test/Perplexity": test_perplexity}, step=epoch) 
        wandb.log({"test/PSNR": test_psnr}, step=epoch)
        if criterion_IFL is not None:
            wandb.log({"test/IFL_loss": test_IFL_loss}, step=epoch)

    print_string = f"=> Test Epoch {epoch}: Total loss: {test_loss:.4f}, Recon loss: {test_recon:.4f}, Perplexity: {test_perplexity:.4f}, PSNR: {test_psnr:.4f}"
    if criterion_IFL is not None:
        print_string += f' IFL: {test_IFL_loss:.4f}'
    print(print_string)
    
    return test_loss

def count_lora_rank_importance(model, train_loader, always_use_this_rank=None):
    with torch.no_grad():
        model.eval()
        pbar = tqdm(train_loader, desc='Calculate Importance', ncols=120)
        for batch_idx, (data, label) in enumerate(pbar):
        
            data = data.to(args.device)
            label = label.to(args.device)
            # target = data.clone()
            assert always_use_this_rank is not None, "always_use_this_rank must be set"
            model.apply(lambda m: set_rank_plan_to_layer(m, always_use_this_rank))
            _, _, _ = model(data)
            pbar_string = f'Batch idx: {batch_idx}/{len(train_loader)}'
            pbar.set_postfix_str(
                pbar_string
            )
        pbar.close()

def threshold_difference_data(difference_data, threshold_ratio=0.9):
    """
    threshold the difference data
    Args:
        difference_data: input difference data tensor [B, C, H, W]
        threshold_ratio: threshold ratio, default 0.8 means keep the top 20% of the difference values
    Returns:
        processed difference data tensor
    """
    # flatten the data
    flat_diff = difference_data.view(difference_data.size(0), -1)
    
    # calculate the threshold for each sample
    thresholds = torch.quantile(flat_diff, threshold_ratio, dim=1)
    
    # extend the threshold to the same shape as the original data
    thresholds = thresholds.view(-1, 1, 1, 1)
    
    # apply the threshold
    difference_data = torch.where(difference_data < thresholds, 
                                torch.zeros_like(difference_data), 
                                difference_data)
    return difference_data

def show_comparison_results(model, draw_loader, epoch, rank_plan):
    model.eval()
    pbar = tqdm(draw_loader, desc='Presenting', ncols=120)
    if rank_plan != None:
        model.apply(lambda m: set_rank_plan_to_layer(m, rank_plan))
    with torch.no_grad():
        for data, label in pbar:
            if data.device != args.device:
                data = data.to(args.device)
                label = label.to(args.device)
            vq_loss, data_recon, perplexity = model(data)
            draw_example_images(data, label, data_recon, epoch, rank_plan)
    pbar.close()
    
def draw_example_images(input_data, target_data, recon_data, epoch, rank_plan):
    input_data_ = input_data.clone().cpu()
    target_data_ = target_data.clone().cpu()
    recon_data_ = recon_data.clone().cpu()
    batch_size = input_data_.shape[0]
    
    # adjust the image data range from [-1,1] to [0,1]
    input_data_ = (input_data_ + 1) / 2
    target_data_ = (target_data_ + 1) / 2
    recon_data_ = (recon_data_ + 1) / 2
    
    # create the image dictionary for wandb
    log_dict = {}
    for i in range(min(batch_size, 3)):  # limit the maximum number of images to 3
        # create the image dictionary, containing the input, target and reconstructed images
        if rank_plan==None:
            log_dict.update({
                f'Img_Input/Img_{i}': wandb.Image(input_data_[i]),
                f'Img_Target/Img_{i}': wandb.Image(target_data_[i]),
                f'Img_NonAligned/Img_{i}': wandb.Image(recon_data_[i])
            })
        else:
            log_dict[f'Img_DK_L{rank_plan}/Img_{i}'] = wandb.Image(recon_data_[i])
    
    # use wandb to record the images
    wandb.log(log_dict, step=epoch)

def test_anomaly_detection(model_anomaly_recon, model_anomaly_erase, data_loader, writer, rank_plan=None):
    model_anomaly_recon.eval()
    model_anomaly_erase.eval()
    flag_first_batch = True
    pbar = tqdm(data_loader, desc='Testing', ncols=120)
    
    if rank_plan != None:
        model_anomaly_erase.apply(lambda m: set_rank_plan_to_layer(m, rank_plan))
        if not getattr(args, "use_preTrain_as_recon", False):
            model_anomaly_recon.apply(lambda m: set_rank_plan_to_layer(m, rank_plan))

    for data, label in pbar:
        data = data.to(args.device)
        label = label.to(args.device)
        _, recon_data, _ = model_anomaly_recon(data)
        _, erase_data, _ = model_anomaly_erase(data)
        difference_data = torch.abs(recon_data - erase_data)
        target_difference_data = torch.abs(label - data)
        mean_target = torch.mean(target_difference_data)
        std_target = torch.std(target_difference_data)
        difference_data = (difference_data - torch.mean(difference_data)) / torch.std(difference_data)
        difference_data = (difference_data + mean_target) * std_target
        difference_data = threshold_difference_data(difference_data)
        if flag_first_batch:
            draw_example_images(data, target_difference_data, difference_data, 0, writer, flag_draw_histogram=True)
            print(f"!!! Draw example comparison images")
            flag_first_batch = False

