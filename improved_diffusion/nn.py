"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from .types_ import *
# from .unet import *


class GaussianConvEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 num_vars=4,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = th.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.in_channels = in_channels
        self.num_vars = num_vars

        modules = []
        if hidden_dims is None:
            if self.num_vars == 4:
                hidden_dims = [16, 32, 32, 64, 64, 128]
            elif self.num_vars == 2:
                hidden_dims = [16, 32, 64, 128]
                
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
    
    def gaussian_parameters(self, h, dim=-1):
        """
        Converts generic real-valued representations into mean and variance
        parameters of a Gaussian distribution

        Args:
            h: tensor: (batch, ..., dim, ...): Arbitrary tensor
            dim: int: (): Dimension along which to split the tensor for mean and
                variance

        Returns:z
            m: tensor: (batch, ..., dim / 2, ...): Mean
            v: tensor: (batch, ..., dim / 2, ...): Variance
        """
        m, h = th.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        
        return m, v


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = th.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        log_var = F.softplus(log_var) + 1e-8
        
        return [mu, log_var]
    
    


class GaussianConvEncoderClf(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 num_vars=4,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = th.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.in_channels = in_channels
        self.num_vars = num_vars

        modules = []
        if hidden_dims is None:
            if self.num_vars == 4:
                hidden_dims = [16, 32, 32, 64, 64, 128]
            elif self.num_vars == 2:
                hidden_dims = [16, 32, 64, 128]
                
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
        
        self.fc = nn.Linear(hidden_dims[-1]*4, 1)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
    
    def gaussian_parameters(self, h, dim=-1):
        """
        Converts generic real-valued representations into mean and variance
        parameters of a Gaussian distribution

        Args:
            h: tensor: (batch, ..., dim, ...): Arbitrary tensor
            dim: int: (): Dimension along which to split the tensor for mean and
                variance

        Returns:z
            m: tensor: (batch, ..., dim / 2, ...): Mean
            v: tensor: (batch, ..., dim / 2, ...): Variance
        """
        m, h = th.split(h, h.size(dim) // 2, dim=dim)
        v = F.softplus(h) + 1e-8
        
        return m, v


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = th.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        log_var = F.softplus(log_var) + 1e-8
        
        return [mu, log_var]
    
    def forward(self, x):
        result = self.encoder(x)
        result = th.flatten(result, start_dim=1)
        
        out = self.fc(result)
        
        return out
    
    


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, latent_dim, num_var):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_var = num_var

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim // self.num_var, self.latent_dim),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim, self.latent_dim // self.num_var)
        )

    def forward(self, x):
        return self.net(x)
    


class CausalModeling(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 num_var=None,
                 learn = False,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_var = num_var

        if learn:
            self.a = th.zeros(self.num_var, self.num_var)
            self.A = nn.Parameter(self.a)
        else:
            self.A = th.tensor([[0, 1], [0, 0]])

        self.nonlinearities = nn.ModuleDict()

        for i in range(self.num_var):
            self.nonlinearities[str(i)] = MLP(latent_dim=latent_dim, num_var=num_var)
        
        # self.nonlinearity1 = nn.Sequential(
        #     nn.Linear(self.latent_dim // self.num_var, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, self.latent_dim // self.num_var)
        # )

        # self.nonlinearity2 = nn.Sequential(
        #     nn.Linear(self.latent_dim // self.num_var, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, self.latent_dim // self.num_var)
        # )

        # self.nonlinearity3 = nn.Sequential(
        #     nn.Linear(self.latent_dim // self.num_var, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, self.latent_dim // self.num_var)
        # )
        
        # self.nonlinearity4 = nn.Sequential(
        #     nn.Linear(self.latent_dim // self.num_var, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, self.latent_dim // self.num_var)
        # )

    def causal_masking(self, u, A):
        u = u.reshape(-1, self.num_var, self.latent_dim // self.num_var)

        z_pre = th.matmul(A.t().to(u.device), u)
        
        return z_pre

    def nonlinearity_add_back_noise(self, u, z_pre):
        # z_pre = z_pre.reshape(-1, self.num_var, self.latent_dim // self.num_var)
        u = u.reshape(-1, self.num_var, self.latent_dim // self.num_var)
        z_post = th.zeros(u.shape)

        for i in range(self.num_var):
            z_post[:, i, :] = self.nonlinearities[str(i)](z_pre[:, i, :]) + u[:, i, :]

        # for i in range(self.num_var):
        # z_post[:, 0, :] = self.nonlinearity1(z_pre[:, 0, :]) + u[:, 0, :]
        # z_post[:, 1, :] = self.nonlinearity2(z_pre[:, 1, :]) + u[:, 1, :]
        # z_post[:, 2, :] = self.nonlinearity3(z_pre[:, 2, :]) + u[:, 2, :]
        # z_post[:, 3, :] = self.nonlinearity4(z_pre[:, 3, :]) + u[:, 3, :]


        return z_post.reshape(-1, self.num_var*(self.latent_dim // self.num_var))


# class MLP_MASK(nn.Module):
#     """ a simple 4-layer MLP """

#     def __init__(self, nin, nout, nh):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(nin, nh),
#             nn.ReLU(),
#             nn.Linear(nh, nh),
#             nn.ReLU(),
#             nn.Linear(nh, nout),
#             nn.Sigmoid(),
#         )
#     def forward(self, x, mask):
#         return self.net(x * mask)
    
    
# self.s_cond = nn.ModuleDict()
# self.t_cond = nn.ModuleDict()

# for i in range(self.dim):
#     self.s_cond[str(i)] = net_class(self.dim*self.k, self.k, nh)

# for i in range(self.dim):
#     self.t_cond[str(i)] = net_class(self.dim*self.k, self.k, nh)
    
# Causal Normalizing Flow
class MultivariateCausalFlow(nn.Module):
    def __init__(self, dim, k, nh=100):
        super().__init__()
        self.dim = dim
        self.k = k


        # self.s_cond = net_class(self.dim*self.k, self.k, 100)
        self.s_cond = nn.Sequential(
                nn.Linear(self.dim*self.k, nh),
                nn.ReLU(),
                nn.Linear(nh, nh),
                nn.ReLU(),
                nn.Linear(nh, self.k),
                nn.Sigmoid(),
            )
        # self.t_cond = net_class(self.dim*self.k, self.k, 100)
        self.t_cond = nn.Sequential(
                nn.Linear(self.dim*self.k, nh),
                nn.ReLU(),
                nn.Linear(nh, nh),
                nn.ReLU(),
                nn.Linear(nh, self.k),
                nn.Sigmoid(),
            )

    def flow(self, e, C):
        e = e.reshape(-1, 2, 256)
        total_dims = e.shape[1]*e.shape[2]
        log_det = th.zeros(e.size(0)).to(e.device)
        # p_logprob = th.zeros(e.size(0)).to(e.device)
        batch_size = e.shape[0]
        z = th.zeros(e.shape).to(e.device)
        
        
        for i in range(self.dim):    
            if 1 in C[:, i]: # does it have any parents (z_3)
                mask = C[:, i].repeat(self.k, 1).T.reshape(total_dims).to(e.device)
            elif 1 not in C[:, i]: # doesnt have parents
                mask = th.zeros(total_dims).to(e.device)
            
            # compute slope and offset
            s = self.s_cond(z.reshape(-1, total_dims) * mask).reshape(batch_size, self.k) # slope
            t = self.t_cond(z.reshape(-1, total_dims) * mask).reshape(batch_size, self.k) # offset

            # slope and offset transformation (affine transformation)
            z[:, i, :] = th.exp(s) * e[:, i, :].reshape(batch_size, self.k) + t
  
            log_det += th.sum(s, dim=1) # dz / de
            

        return [z.reshape(-1, 512), log_det]
    
    def reverse(self, z, C):
        z = z.reshape(-1, 2, 256)
        prior = MultivariateNormal(th.ones(z.shape[1]*z.shape[2]).to(z.device), th.eye(z.shape[1]*z.shape[2]).to(z.device))
        total_dims = z.shape[1]*z.shape[2]
        log_det = th.zeros(z.size(0)).to(z.device)
        # p_logprob = th.zeros(z.size(0)).to(z.device)
        batch_size = z.shape[0]
        e = th.zeros(batch_size, self.dim, self.k).to(z.device)
        
        
        for i in range(self.dim):
            
            if 1 in C[:, i]: # does it have any parents (z_3)
                # mask = self.C[:, i].reshape(self.dim).to(device) # [1, 1, 0, 0]
                mask = C[:, i].repeat(self.k, 1).T.reshape(total_dims).to(e.device)
            elif 1 not in C[:, i]: # doesnt have parents
                mask = th.zeros(total_dims).to(e.device)
            
            # compute slope and offset
            s = self.s_cond(z.reshape(-1, total_dims) * mask).reshape(batch_size, self.k) # slope
            t = self.t_cond(z.reshape(-1, total_dims) * mask).reshape(batch_size, self.k) # offset
            
        
            # slope and offset transformation (affine transformation)
            e[:, i, :] = th.exp(-s) * (z[:, i, :].reshape(batch_size, self.k) - t)

            log_det -= th.sum(s, dim=1) # dz / de

        
        p_log_prob = prior.log_prob(e.reshape(-1, z.shape[1]*z.shape[2]))
            
        return [log_det, p_log_prob]
    

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def kl_normal(qm, qv, pm, pv):
        """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension

        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance

        Return:
            kl: tensor: (batch,): kl between each sample
        """
        element_wise = 0.5 * (th.log(pv) - th.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)

        return kl


def reparameterize(m, v):
    """
    Reparameterization Trick.
    """
    sample = th.randn(m.size()).to(m.device)
    z = m + (v**0.5)*sample
    
    return z


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
