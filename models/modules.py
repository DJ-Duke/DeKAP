import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from source.args import args as pargs

StandardConv = nn.Conv2d
StandardBN = nn.BatchNorm2d

def weight_to_matrix(weight, weight_type, force_v2=False):
    """
    
    Args:
        weight: shape (ch_out, ch_in, kernel_len, kernel_len)
        
    Returns:
        matrix: shape (ch_in * kernel_len, ch_out * kernel_len)
    """
    if force_v2:
        version = 2
    else:
        version = getattr(pargs, "lora_weight_org_ver", 1)
    assert version in [1, 2], "lora_weight_org_ver must be 1 or 2"
        
    if weight_type == "conv":
        if version == 1:
            ch_out, ch_in, kernel_len, kernel_len = weight.shape
            matrix = weight.view(ch_out* kernel_len, ch_in* kernel_len)
        elif version == 2:
            ch_out, ch_in, kernel_len, kernel_len = weight.shape
            matrix = weight.view(ch_out, ch_in * kernel_len * kernel_len)
    elif weight_type == "deconv":
        if version == 1:    
            ch_in, ch_out, kernel_len, kernel_len = weight.shape
            matrix = weight.view(ch_out * kernel_len, ch_in * kernel_len)
        elif version == 2:
            ch_in, ch_out, kernel_len, kernel_len = weight.shape
            matrix = weight.view(ch_in, ch_out * kernel_len * kernel_len)
    elif weight_type == "embed":
        matrix = weight
    else:
        raise ValueError("weight_type must be 'conv' or 'deconv' or 'embed'")
    
    return matrix

def obtain_full_ft_sigvalue_func(fullft_changes, weight_type, layer_name='', original_parameter=None, parameter_per_rank=None):
    weight_shape = fullft_changes.shape
    num_params = fullft_changes.numel()
    weight_matrix = weight_to_matrix(fullft_changes.clone().detach(), weight_type)
    if pargs.lora_weight_org_ver == 1:
        if weight_type in ["conv", "deconv"]:
            kernel_size = weight_shape[2]
            parameter_per_rank = (weight_shape[0] + weight_shape[1]) * kernel_size
        elif weight_type == "embed":
            parameter_per_rank = (weight_shape[0] + weight_shape[1])
    num_params = weight_matrix.numel()
    _, S, _ = torch.svd(weight_matrix)
    sigvalue_vec = S.cpu().numpy()
    
    sigvalue_magnitude = sigvalue_vec
    cumsum_sigvalue_magnitude = np.cumsum(sigvalue_magnitude)
    cumsum_sigvalue_magnitude = np.insert(cumsum_sigvalue_magnitude, 0, 1e-1)
    cumsum_sigvalue_magnitude = cumsum_sigvalue_magnitude[:-1]
    sigvalue_magnitude_ratio = sigvalue_magnitude / cumsum_sigvalue_magnitude
    sigvalue_vec = sigvalue_magnitude_ratio 
    sigvalue_vec[0] += 20
    
    
    maximum_rank = min(len(sigvalue_vec), num_params / parameter_per_rank)
    if maximum_rank == len(sigvalue_vec):
        parameter_per_rank_vec = np.ones(len(sigvalue_vec)) * parameter_per_rank
    elif maximum_rank < len(sigvalue_vec):
        # obtain the ceiling of maximum_rank
        maximum_rank = int(np.ceil(maximum_rank))
        maximum_rank_minus_1 = maximum_rank - 1
        parameter_per_rank_vec = np.ones(maximum_rank) * parameter_per_rank
        parameter_per_rank_vec[-1] = num_params - (maximum_rank_minus_1 * parameter_per_rank)
        sigvalue_vec_original = sigvalue_vec
        sigvalue_vec = sigvalue_vec[:maximum_rank]
        # calcualte the last effective singular value
        last_effective_sigvalue = np.sqrt(np.sum(sigvalue_vec_original[maximum_rank:]**2))
        last_effective_sigvalue = min((last_effective_sigvalue, sigvalue_vec[maximum_rank_minus_1-1]-1e-6))
        sigvalue_vec[-1] = last_effective_sigvalue
    
    return sigvalue_vec, parameter_per_rank_vec, int(maximum_rank)
        
class NonAffineNoStatsBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineNoStatsBN, self).__init__(
            dim, affine=False, track_running_stats=False
        )

class MultitaskMaskConvChange(nn.Conv2d):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if pargs.pre_train == 0: # Default true
            self.weight.requires_grad = False

        self.with_lora_change = getattr(pargs, "with_lora_change", False)
        if self.with_lora_change:
            self.lora_rank = pargs.lora_rank
            self.lora_rank_specific = self.lora_rank
            self.lora_alpha = pargs.lora_alpha
            self.lora_scaling = self.lora_alpha 
            self.full_rank = False

    def add_changes(self):
        # the changes store the expert knowledge for each task
        self.flag_with_changes = True
        self.changes = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(self.weight))
                for _ in range(pargs.num_ft_changes)
            ]
        )
    def reinit_changes(self): 
        for j in range(pargs.num_ft_changes):
            self.changes[j].data = torch.zeros_like(self.changes[j].data) 
            if self.with_lora_change:
                nn.init.kaiming_uniform_(self.lora_A[j], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[j])

    def backup_changes(self):
        # backup the current changes parameters
        self.backup_changes = [
            self.changes[j].data.clone().detach()
            for j in range(pargs.num_ft_changes)
        ]
    
    def restore_changes(self):
        # restore the changes parameters from the backup
        for j in range(pargs.num_ft_changes):
            self.changes[j].data = self.backup_changes[j]
        
    def forward(self, x):
        if getattr(self, "pretrain", False):
            w = self.weight
            x = F.conv2d(
                x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return x

        if getattr(self, "using_backup_changes", False):
            w = self.weight + self.backup_changes[self.change_idx]
            x = F.conv2d(
                x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return x
        
        if self.with_lora_change:
            if self.max_alloc_rank == 0 or self.target_rank_to_use == 0:
                lora_changes = 0
            else:
                sel_index = self.target_rank_to_use
                lora_A_tensors = [self.lora_A_list[i] for i in range(sel_index)]
                lora_B_tensors = [self.lora_B_list[i] for i in range(sel_index)]
                lora_A_cat = torch.cat(lora_A_tensors, dim=0)
                lora_B_cat = torch.cat(lora_B_tensors, dim=1)
                lora_changes = (lora_B_cat @ lora_A_cat).view(self.weight.shape) * self.lora_scaling
        
        if getattr(self, "flag_with_changes", False):
            w = self.weight + lora_changes

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        
        return x

    def __repr__(self):
        return f"ConvChange({self.in_channels}, {self.out_channels})"
    
    def obtain_full_ft_sigvalue(self):
        fullft_changes = self.backup_changes[self.change_idx].data
        weight_type = 'conv'
        sigvalue_vec, parameter_per_rank_vec, maximum_rank = obtain_full_ft_sigvalue_func(fullft_changes, weight_type, self.module_name, self.weight, self.prop_parameter_per_rank)
        self.fullft_sigvalue_vec = sigvalue_vec
        self.parameter_per_rank = parameter_per_rank_vec
        self.maximum_rank = maximum_rank

    @property
    def prop_parameter_per_rank(self):
        assert pargs.lora_weight_org_ver == 2, "parameter_per_rank is not supported for lora_weight_org_ver != 2"
        weight_shape = self.weight.shape
        kernel_size = weight_shape[2] 
        in_channels = weight_shape[1]
        out_channels = weight_shape[0]
        return (in_channels * kernel_size * kernel_size + out_channels)
    
    def add_new_rank_list(self, max_alloc_rank):
        # this function is to directly allocate the largest rank, but can choose how many ranks to actually use
        self.flag_given_rank_list = True 
        weight_shape = self.weight.shape
        kernel_size = weight_shape[2] 
        in_channels = weight_shape[1]
        out_channels = weight_shape[0]
        self.max_alloc_rank = max_alloc_rank
        self.lora_A_list = nn.ParameterList(
            [
                nn.Parameter(self.weight.new_zeros((1, in_channels * kernel_size * kernel_size)))
                for j in range(max_alloc_rank)
            ]
        )
        self.lora_B_list = nn.ParameterList(
            [
                nn.Parameter(self.weight.new_zeros((out_channels, 1)))
                for j in range(max_alloc_rank)
            ]
        )
        for j in range(max_alloc_rank):
            nn.init.kaiming_uniform_(self.lora_A_list[j], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[j])
        
        self.has_added_rank = False

    def reinit_whole_rank_list(self):
        
        for i in range(self.max_alloc_rank):
            nn.init.kaiming_uniform_(self.lora_A_list[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[i])

    def set_rank_list_usage(self, target_rank=0):
        self.target_rank_to_use = target_rank
        if self.max_alloc_rank == 0:
            self.lora_param_num = 0
        else:
            self.lora_param_num = (self.lora_A_list[0].numel() + self.lora_B_list[0].numel()) * target_rank
    def set_target_rank_plan(self, plan_list):
        self.target_rank_plan_list = plan_list
    
    def set_rank_plan(self, rank_plan_idx):
        self.target_rank_to_use = self.target_rank_plan_list[rank_plan_idx]

class MultitaskMaskDeConvChange(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if pargs.pre_train == 0:  # Default true
            self.weight.requires_grad = False

        self.with_lora_change = getattr(pargs, "with_lora_change", False)
        if self.with_lora_change:
            self.lora_rank = pargs.lora_rank
            self.lora_rank_specific = self.lora_rank
            self.lora_alpha = pargs.lora_alpha
            self.lora_scaling = self.lora_alpha #/ (self.lora_rank+1e-6)
            self.full_rank = False

    
    def add_changes(self):
        self.flag_with_changes = True
        self.changes = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(self.weight))
                for _ in range(pargs.num_ft_changes)
            ]
        )
        
    def backup_changes(self):
        """备份当前的changes参数"""
        self.backup_changes = [
            self.changes[j].data.clone().detach()
            for j in range(pargs.num_ft_changes)
        ]

    def reinit_changes(self): 
        for j in range(pargs.num_ft_changes):
            self.changes[j].data = torch.zeros_like(self.changes[j].data) 
            if self.with_lora_change:
                nn.init.kaiming_uniform_(self.lora_A[j], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[j])

    def restore_changes(self):
        # restore the changes parameters from the backup
        for j in range(pargs.num_ft_changes):
            self.changes[j].data = self.backup_changes[j]


    def forward(self, x):
        if getattr(self, "pretrain", False):
            w = self.weight
            x = F.conv_transpose2d(
                x, w, self.bias, self.stride, self.padding,
                output_padding=self.output_padding,  
                groups=self.groups, dilation=self.dilation
            )
            return x
        
        if getattr(self, "using_backup_changes", False):
            w = self.weight + self.backup_changes[self.change_idx]
            x = F.conv_transpose2d(
                x, w, self.bias, self.stride, self.padding,
                output_padding=self.output_padding, 
                groups=self.groups, dilation=self.dilation
            )
            return x
        
        if self.with_lora_change:
            if self.max_alloc_rank == 0 or self.target_rank_to_use == 0:
                lora_changes = 0
            else:
                sel_index = self.target_rank_to_use
                lora_A_tensors = [self.lora_A_list[i] for i in range(sel_index)]
                lora_B_tensors = [self.lora_B_list[i] for i in range(sel_index)]
                lora_A_cat = torch.cat(lora_A_tensors, dim=0)
                lora_B_cat = torch.cat(lora_B_tensors, dim=1)
                lora_changes = (lora_B_cat @ lora_A_cat).view(self.weight.shape) * self.lora_scaling

        if getattr(self, "flag_with_changes", False):
            w = self.weight + lora_changes

        x = F.conv_transpose2d(
            x, w, self.bias, self.stride, self.padding,
            output_padding=self.output_padding, 
            groups=self.groups, dilation=self.dilation
        )
        
        return x

    def __repr__(self):
        return f"MultitaskMaskDeConvChange({self.in_channels}, {self.out_channels})"

    def obtain_full_ft_sigvalue(self):
        fullft_changes = self.backup_changes[self.change_idx].data
        weight_type = 'deconv'
        sigvalue_vec, parameter_per_rank_vec, maximum_rank = obtain_full_ft_sigvalue_func(fullft_changes, weight_type, self.module_name, self.weight, self.prop_parameter_per_rank)
        self.fullft_sigvalue_vec = sigvalue_vec
        self.parameter_per_rank = parameter_per_rank_vec
        self.maximum_rank = maximum_rank

    @property
    def prop_parameter_per_rank(self):
        assert pargs.lora_weight_org_ver == 2, "parameter_per_rank is not supported for lora_weight_org_ver != 2"
        weight_shape = self.weight.shape
        kernel_size = weight_shape[2] 
        in_channels = weight_shape[0]
        out_channels = weight_shape[1]
        return (in_channels + out_channels* kernel_size* kernel_size)
    
    def add_new_rank_list(self, max_alloc_rank):
        self.flag_given_rank_list = True 
        weight_shape = self.weight.shape
        kernel_size = weight_shape[2] 
        in_channels = weight_shape[0]
        out_channels = weight_shape[1]
        self.max_alloc_rank = max_alloc_rank
        self.lora_A_list = nn.ParameterList(
            [
                nn.Parameter(self.weight.new_zeros((1,  out_channels * kernel_size * kernel_size)))
                for j in range(max_alloc_rank)
            ]
        )
        self.lora_B_list = nn.ParameterList(
            [
                nn.Parameter(self.weight.new_zeros((in_channels , 1)))
                for j in range(max_alloc_rank)
            ]
        )
        for j in range(max_alloc_rank):
            nn.init.kaiming_uniform_(self.lora_A_list[j], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[j])
        
        self.has_added_rank = False
    
    def reinit_whole_rank_list(self):
        
        for i in range(self.max_alloc_rank):
            nn.init.kaiming_uniform_(self.lora_A_list[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[i])

    def set_rank_list_usage(self, target_rank=0):
        self.target_rank_to_use = target_rank
        if self.max_alloc_rank == 0:
            self.lora_param_num = 0
        else:
            self.lora_param_num = (self.lora_A_list[0].numel() + self.lora_B_list[0].numel()) * target_rank
    
    def set_target_rank_plan(self, plan_list):
        self.target_rank_plan_list = plan_list
    
    def set_rank_plan(self, rank_plan_idx):
        self.target_rank_to_use = self.target_rank_plan_list[rank_plan_idx]


class MultitaskMaskEmbeddingChange(nn.Embedding): 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if pargs.pre_train == 0: # Default true
            self.weight.requires_grad = False

        self.with_lora_change = getattr(pargs, "with_lora_change", False)
        if self.with_lora_change:
            self.lora_rank = pargs.lora_rank
            self.lora_rank_specific = self.lora_rank
            self.lora_alpha = pargs.lora_alpha
            self.lora_scaling = self.lora_alpha # / (self.lora_rank+1e-6)
            self.full_rank = False


    def add_changes(self):
        self.flag_with_changes = True
        self.changes = nn.ParameterList(
            [
                nn.Parameter(torch.zeros_like(self.weight))
                for _ in range(pargs.num_ft_changes)
            ]
        )

    def backup_changes(self):
        self.backup_changes = [
            self.changes[j].data.clone().detach()
            for j in range(pargs.num_ft_changes)
        ]

    def reinit_changes(self): # 基于LTH应当从初始点恢复的思想， 所以changes应当从0开始。
        for j in range(pargs.num_ft_changes):
            self.changes[j].data = torch.zeros_like(self.changes[j].data) #self.backup_changes[j].data.clone().detach()
            if self.with_lora_change: 
                nn.init.kaiming_uniform_(self.lora_A[j], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[j])
            
    def restore_changes(self):
        for j in range(pargs.num_ft_changes):
            self.changes[j].data = self.backup_changes[j]

    
    @property
    def my_weight(self):
        if getattr(self, "pretrain", False):
            return self.weight
        
        if getattr(self, "using_backup_changes", False):
            w = self.weight + self.backup_changes[self.change_idx]
            return w
        
        if self.with_lora_change:
            if self.max_alloc_rank == 0 or self.target_rank_to_use == 0:
                lora_changes = 0
            else:
                sel_index = self.target_rank_to_use
                lora_A_tensors = [self.lora_A_list[i] for i in range(sel_index)]
                lora_B_tensors = [self.lora_B_list[i] for i in range(sel_index)]
                lora_A_cat = torch.cat(lora_A_tensors, dim=0)
                lora_B_cat = torch.cat(lora_B_tensors, dim=1)
                lora_changes = (lora_B_cat @ lora_A_cat).view(self.weight.shape) * self.lora_scaling

        if getattr(self, "flag_with_changes", False):
            w = self.weight + lora_changes

        return w

    def __repr__(self):
        return f"MultitaskMaskEmbeddingChange({self._num_embeddings}, {self._embedding_dim})"
    
    def obtain_full_ft_sigvalue(self):
        fullft_changes = self.backup_changes[self.change_idx].data
        weight_type = 'embed'
        sigvalue_vec, parameter_per_rank_vec, maximum_rank = obtain_full_ft_sigvalue_func(fullft_changes, weight_type, self.module_name, self.weight, self.prop_parameter_per_rank)
        self.fullft_sigvalue_vec = sigvalue_vec
        self.parameter_per_rank = parameter_per_rank_vec
        self.maximum_rank = maximum_rank

    @property
    def prop_parameter_per_rank(self):
        weight_shape = self.weight.shape
        num_embeddings = weight_shape[0]
        embedding_dim = weight_shape[1]
        return (num_embeddings + embedding_dim)
    

    def add_new_rank_list(self, max_alloc_rank):
        self.flag_given_rank_list = True 
        weight_shape = self.weight.shape
        num_embeddings = weight_shape[0]
        embedding_dim = weight_shape[1]
        self.max_alloc_rank = max_alloc_rank
        self.lora_A_list = nn.ParameterList(
            [
                nn.Parameter(self.weight.new_zeros((1, embedding_dim)))
                for j in range(max_alloc_rank)
            ]
        )
        self.lora_B_list = nn.ParameterList(
            [
                nn.Parameter(self.weight.new_zeros((num_embeddings, 1)))
                for j in range(max_alloc_rank)
            ]
        )
        for j in range(max_alloc_rank):
            nn.init.kaiming_uniform_(self.lora_A_list[j], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[j])
        
        self.has_added_rank = False

    def reinit_whole_rank_list(self):
        if getattr(pargs, "using_svd_to_init_lora", False):
            with torch.no_grad():
                U, S, V = torch.svd(self.backup_changes[0].data.clone())
                for j in range(self.max_alloc_rank):
                    self.lora_A_list[j].data = torch.transpose(V[:,j:j+1] * torch.sqrt(S[j]), 0, 1)
                    self.lora_B_list[j].data = U[:,j:j+1] * torch.sqrt(S[j])
        else:
            for i in range(self.max_alloc_rank):
                nn.init.kaiming_uniform_(self.lora_A_list[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_list[i])

    def set_rank_list_usage(self, target_rank=0):
        self.target_rank_to_use = target_rank
        if self.max_alloc_rank == 0:
            self.lora_param_num = 0
        else:
            self.lora_param_num = (self.lora_A_list[0].numel() + self.lora_B_list[0].numel()) * target_rank
    
    def set_target_rank_plan(self, plan_list):
        self.target_rank_plan_list = plan_list
    
    def set_rank_plan(self, rank_plan_idx):
        self.target_rank_to_use = self.target_rank_plan_list[rank_plan_idx]

