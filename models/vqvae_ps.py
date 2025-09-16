import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.nn import functional as F
from . import modules

from source.args import args
from .builder import Builder

def conv1x1(in_planes, out_planes, builder, stride=1, bias=None):
    return builder.conv1x1(in_planes, out_planes, stride=stride, bias=bias)

def conv3x3(in_planes, out_planes, builder, stride=1, bias=None):
    return builder.conv3x3(in_planes, out_planes, stride=stride, bias=bias)

def conv4x4(in_planes, out_planes, builder, stride=1, bias=None):
    return builder.conv4x4(in_planes, out_planes, stride=stride, bias=bias)

def conv5x5(in_planes, out_planes, builder, stride=1, bias=None):
    return builder.conv5x5(in_planes, out_planes, stride=stride, bias=bias)

def conv7x7(in_planes, out_planes, builder, stride=1, bias=None):
    return builder.conv7x7(in_planes, out_planes, stride=stride, bias=bias)

def deconv3x3(in_planes, out_planes, builder, stride=1, output_padding=1, bias=None):
    return builder.deconv3x3(in_planes, out_planes, stride=stride, output_padding=output_padding, bias=bias)

def deconv4x4(in_planes, out_planes, builder, stride=1, output_padding=1, bias=None):
    return builder.deconv4x4(in_planes, out_planes, stride=stride, output_padding=output_padding, bias=bias)

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, builder):
        super(Residual, self).__init__()
        self.relu1 = nn.ReLU(True)
        self.layer1_conv = conv3x3(in_channels, num_residual_hiddens, builder, stride=1, bias=False)
        self.relu2 = nn.ReLU(True)
        self.layer2_conv = conv1x1(num_residual_hiddens, num_hiddens, builder, stride=1, bias=False)
    
    def forward(self, x):
        out = self.relu1(x)
        out = self.layer1_conv(out)
        out = self.relu2(out)
        out = self.layer2_conv(out)
        return x + out

class Residual_Equalizer(nn.Module):
    def __init__(self, dim_embed, num_layers):
        super(Residual_Equalizer, self).__init__()
        self._layers = nn.ModuleList([nn.Conv2d(dim_embed, dim_embed, kernel_size=1, stride=1, bias=False) for _ in range(num_layers)])

    def forward(self, x):
        out = x
        for layer in self._layers:
            out = layer(out)
            out = F.relu(out)
        return x + out


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, builder):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens, builder)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class VQVAE_encoder(nn.Module):
    def __init__(self, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32):
        super(VQVAE_encoder, self).__init__()

        builder = Builder()
        activation = nn.ReLU(True)

        
        self.layer1_conv = conv4x4(3, num_hiddens // 2, builder, stride=2)
        self.relu1 = activation
        
        self.layer2_conv = conv4x4(num_hiddens // 2, num_hiddens, builder, stride=2)
        self.relu2 = activation
        
        self.layer3_conv = conv3x3(num_hiddens, num_hiddens, builder, stride=1)
        self.relu3 = activation

        self.resblk = ResidualStack(in_channels=num_hiddens,
                                    num_hiddens=num_hiddens,
                                    num_residual_layers=num_residual_layers,
                                    num_residual_hiddens=num_residual_hiddens, 
                                    builder=builder)
        
    def forward(self, x):
        x = self.layer1_conv(x)
        x = self.relu1(x)
        
        x = self.layer2_conv(x)
        x = self.relu2(x)
        
        x = self.layer3_conv(x)
        x = self.resblk(x)
        return x

class VQVAE_decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers=2, num_residual_hiddens=32):
        super(VQVAE_decoder, self).__init__()

        builder = Builder()
        activation = nn.ReLU(True)

        # 保持高分辨率特征处理用普通卷积
        self.layer1_conv = conv3x3(in_channels, num_hiddens, builder, stride=1)
        
        self.resblk = ResidualStack(in_channels=num_hiddens,
                        num_hiddens=num_hiddens,
                        num_residual_layers=num_residual_layers,
                        num_residual_hiddens=num_residual_hiddens, 
                        builder=builder)
        
        # 第一次上采样用反卷积（因为特征还比较抽象，可学习性更重要）
        self.layer2_deconv = deconv4x4(num_hiddens, num_hiddens//2, builder, stride=2, output_padding=0)
        self.relu1 = activation
        
        # 第二次上采样用上采样+卷积（因为接近输出，避免棋盘格效应更重要）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.layer3_conv = conv3x3(num_hiddens//2, 3, builder)

    
    def forward(self, z):
        x = self.layer1_conv(z)
        x = self.resblk(x)
        
        # 第一次上采样
        x = self.layer2_deconv(x)
        x = self.relu1(x)
        
        # 第二次上采样
        x = self.upsample(x)
        x = self.layer3_conv(x)
        x = F.sigmoid(x)   
        
        return 2 * x - 1

class VectorQuantizerEMA(nn.Module):
    '''
    Embedding 目前我们不考虑用sparse mask, 其实后面可以改成也需要的。在FT的时候是要Fix住的。不过后面也可以修改成为同样需要改的。不过估计EMA的不行，如果同样需要对Embedding进行修改的话，那么需要用非EMA的才行。
    '''
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5, no_change_vq = False):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        builder = Builder()
        '''
        以均匀分布进行初始化困惑度会更高。训练的也会更快。normal()来说。normal()可能一开始会陷入困惑的的陷阱。
        '''
        # if args.use_my_embedding:
        #     self._embedding = modules.MultitaskMaskEmbeddingChange(self._num_embeddings, self._embedding_dim)
        # else:
        #     self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding = builder.embedding_layer(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        self._decay = decay
        self._epsilon = epsilon
        self.no_change_vq = False
        self.finetune_vq = False
        

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.my_weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.my_weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        flag_count_lora_rank_importance = getattr(self._embedding, "flag_count_lora_rank_importance", False)
        if flag_count_lora_rank_importance:
            temp_z = []
            for i in range(self._embedding.target_rank_to_use):
                sub_lora_changes = self._embedding.my_weight_sublora(i)
                z = torch.matmul(encodings, sub_lora_changes)
                temp_z = torch.abs(z)
                self._embedding.rank_importance_list[i] = (self._embedding.rank_importance_list[i]*self._embedding.num_samples + torch.sum(torch.mean(temp_z, dim=1)))/(self._embedding.num_samples+encodings.shape[0])
            self._embedding.num_samples += encodings.shape[0]
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.my_weight).view(input_shape)
        # quantized = self._embedding(encoding_indices).view(input_shape)
        
        # Use EMA to update the embedding vectors
        q_latent_loss = 0
        if self.training and not self.no_change_vq:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        elif self.training and self.finetune_vq:
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    def output_encoding_indices(self, inputs):
        # 类似forward, 但不是输出查表后的向量，只是输出离散embeddin
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.my_weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.my_weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        input_info = {
            "device": inputs.device,
            "shape": input_shape
        }
        return encoding_indices, input_info

    def input_encoding_indices(self, encoding_indices, input_info):
        device = input_info['device']
        input_shape = input_info['shape']
        
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.my_weight).view(input_shape)
        # quantized = self._embedding(encoding_indices).view(input_shape)
        
        
        # Straight Through Estimator
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class VQVAE_full_ps(nn.Module):
    def __init__(self, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=1024, embedding_dim=128, commitment_cost=0.25, decay=0.99):
        super(VQVAE_full_ps, self).__init__()
        self.name = "VQVAE_ps"

        builder = Builder()

        # num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32
        self.encoder = VQVAE_encoder(num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)

        self.pre_vq_conv = conv1x1(num_hiddens, embedding_dim, builder, stride=1)
        self.vector_quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        
        self.decoder = VQVAE_decoder(in_channels=embedding_dim, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)


    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)


        loss, quantized, perplexity, _ = self.vector_quantizer(z)

        if getattr(self, "flag_with_equalizer", False):
            quantized = self.equalizer(quantized)

        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity
    
    def set_finetune_vq(self, finetune_vq):
        # 在FT的阶段, 是否计算VQ的quantized 到 inputs的损失
        self.vector_quantizer.finetune_vq = finetune_vq

    def set_no_change_vq(self, no_change_vq):
        self.vector_quantizer.no_change_vq = no_change_vq

    
    def adding_equalizer_layer(self, num_layers=3):
        self.flag_with_equalizer = True
        # 使用与模型相同的device
        device = next(self.parameters()).device
        self.equalizer = Residual_Equalizer(args.embedding_dim, num_layers).to(device)
        # 使用 Kaiming 初始化
        for layer in self.equalizer._layers:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            # use zero init
            # layer.weight.data.zero_()
        print(f"Equalizer layer have number of parameters: {sum(p.numel() for p in self.equalizer.parameters())}")

def VQVAE_ps(): # ZFFT中暂时使用的模型
    return VQVAE_full_ps(num_hiddens = args.num_hiddens, num_residual_layers = args.num_residual_layers, num_residual_hiddens = args.num_residual_hiddens, num_embeddings = args.num_embeddings, embedding_dim = args.embedding_dim, commitment_cost = args.commitment_cost, decay = args.decay)