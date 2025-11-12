"""Particle Transformer (ParT)
Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
"""

import math
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import copy

from utils.logger import LOGGER as _logger
from .head import DINOHead


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy**2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx**2))) * sx**2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps) ** (-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split(
        (1, 1, 1), dim=1
    )
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split(
        (1, 1, 1), dim=1
    )

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = (
            ((pti <= ptj) * pti + (pti > ptj) * ptj)
            if for_onnx
            else torch.minimum(pti, ptj)
        )
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(
            dim=1, keepdim=True
        )
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    # assert len(outputs) == num_outputs
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat(
        (
            torch.arange(0, batch_size, device=uu.device)
            .repeat_interleave(num_fts * num_pairs)
            .unsqueeze(0),
            torch.arange(0, num_fts, device=uu.device)
            .repeat_interleave(num_pairs)
            .repeat(batch_size)
            .unsqueeze(0),
            idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
            idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
        ),
        dim=0,
    )
    return torch.sparse_coo_tensor(
        i,
        uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device,
    ).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    @torch.compiler.disable
    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    # q = min(1, random.uniform(*self.target))
                    # maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    q = torch.min(
                        torch.ones(1, device=mask.device),
                        torch.rand(1, device=mask.device)
                        * (self.target[1] - self.target[0])
                        + self.target[0],
                    )[0]
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=False, activation="gelu"):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend(
                [
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                ]
            )
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = x.contiguous()
            x = self.input_bn(x)
            # x: (seq_len, batch, embed_dim)
            x = x.permute(2, 0, 1).contiguous()
        else:
            x = x.permute(2, 0, 1).contiguous()
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
        self,
        pairwise_lv_dim,
        pairwise_input_dim,
        dims,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        mode="sum",
        normalize_input=True,
        activation="gelu",
        eps=1e-8,
        for_onnx=False,
    ):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(
            pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx
        )
        self.out_dim = dims[-1]

        if self.mode == "concat":
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                # module_list.extend(
                #     [
                #         nn.Conv1d(input_dim, dim, 1),
                #         nn.BatchNorm1d(dim) if normalize_input else nn.Identity(),
                #         nn.GELU() if activation == "gelu" else nn.ReLU(),
                #     ]
                # )
                
                module_list.append(nn.Conv1d(input_dim, dim, 1))
                if normalize_input:
                    module_list.append(nn.BatchNorm1d(dim))
                module_list.append(
                    nn.GELU() if activation == "gelu" else nn.ReLU()
                )
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        
        elif self.mode == "sum":
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    # module_list.extend(
                    #     [
                    #         nn.Conv1d(input_dim, dim, 1),
                    #         nn.BatchNorm1d(dim) if normalize_input else nn.Identity(),
                    #         nn.GELU() if activation == "gelu" else nn.ReLU(),
                    #     ]
                    # )
                    
                    module_list.append(nn.Conv1d(input_dim, dim, 1))
                    if normalize_input:
                        module_list.append(nn.BatchNorm1d(dim))
                    module_list.append(
                        nn.GELU() if activation == "gelu" else nn.ReLU()
                    )
                    
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    # module_list.extend(
                    #     [
                    #         nn.Conv1d(input_dim, dim, 1),
                    #         nn.BatchNorm1d(dim) if normalize_input else nn.Identity(),
                    #         nn.GELU() if activation == "gelu" else nn.ReLU(),
                    #     ]
                    # )
                    module_list.append(nn.Conv1d(input_dim, dim, 1))
                    if normalize_input:
                        module_list.append(nn.BatchNorm1d(dim))
                    module_list.append(
                        nn.GELU() if activation == "gelu" else nn.ReLU()
                    )
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError("`mode` can only be `sum` or `concat`")

    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert x is not None or uu is not None
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(
                    seq_len,
                    seq_len,
                    offset=-1 if self.remove_self_pair else 0,
                    device=(x if x is not None else uu).device,
                )
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == "concat":
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == "concat":
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == "sum":
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(
                batch_size,
                self.out_dim,
                seq_len,
                seq_len,
                dtype=elements.dtype,
                device=elements.device,
            )
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        ffn_ratio=4,
        dropout=0.1,
        attn_dropout=0.1,
        activation_dropout=0.1,
        add_bias_kv=False,
        activation="gelu",
        scale_fc=True,
        scale_attn=True,
        scale_heads=True,
        scale_resids=True,
        enable_mem_efficient=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)

        self.enable_mem_efficient = enable_mem_efficient
        if enable_mem_efficient:
            try:
                from .memory_efficient_attention import (
                    MemoryEfficientMultiheadAttention,
                )

                MultiheadAttention = MemoryEfficientMultiheadAttention
            except ImportError as e:
                _logger.error(
                    f"MemoryEfficientMultiheadAttention not available: {e}"
                    "Falling back to standard multihead attention"
                )
                MultiheadAttention = nn.MultiheadAttention
                self.enable_mem_efficient = False
        else:
            _logger.debug("Using standard multihead attention")
            MultiheadAttention = nn.MultiheadAttention
        self.attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = (
            nn.Parameter(torch.ones(num_heads), requires_grad=True)
            if scale_heads
            else None
        )
        self.w_resid = (
            nn.Parameter(torch.ones(embed_dim), requires_grad=True)
            if scale_resids
            else None
        )

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if self.enable_mem_efficient:
            padding_mask = torch.zeros_like(padding_mask, dtype=x.dtype).masked_fill(
                padding_mask, -float("inf")
            )
        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                num_cls_token = x_cls.size(0)
                padding_mask = torch.cat(
                    (torch.zeros_like(padding_mask[:, :num_cls_token]), padding_mask),
                    dim=1,
                )
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(
                x_cls, u, u, key_padding_mask=padding_mask, need_weights=False
            )[
                0
            ]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(
                x,
                x,
                x,
                key_padding_mask=padding_mask,
                attn_mask=attn_mask,
                need_weights=False,
            )[
                0
            ]  # (seq_len, batch, embed_dim)

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum("tbhd,h->tbdh", x, self.c_attn)
            # x = x.permute(0, 1, 3, 2) * self.c_attn.reshape(1, 1, 1, -1)  # rewrite einsum
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = x + residual

        return x




class ParticleTransformer(nn.Module):
    """Particle Transformer (ParT) architecture for processing particle physics data.

    The model takes particle features and their interactions as input and processes them through
    a series of self-attention blocks, followed by class attention blocks to produce final predictions.
    The architecture is illustrated in the paper Figure 3.

    Args:
        input_dim (int): Dimension of input particle features
        proj_dim (int, optional): Number of projection dimensions
        pair_input_dim (int, default=4): Dimension of pairwise kinematic features (dR, kt, z, m)
        pair_extra_dim (int, default=0): Dimension of additional pairwise features
        remove_self_pair (bool, default=False): Whether to remove self-interaction pairs
        use_pre_activation_pair (bool, default=True): Whether to use pre-activation in pair embedding
        embed_dims (list, default=[128, 512, 128]): Dimensions for particle embedding layers
        pair_embed_dims (list, default=[64, 64, 64]): Dimensions for pair embedding layers
        num_heads (int, default=8): Number of attention heads
        num_layers (int, default=8): Number of transformer layers
        num_cls_layers (int, default=2): Number of class attention layers
        num_cls_tokens (int, default=1): Number of class tokens
        block_params (dict, optional): Parameters for transformer blocks
        cls_block_params (dict, default={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0}):
            Parameters for class attention blocks
        head_params (list, optional): List of tuples (dim, dropout) for final FC layers
        norm_proj (bool, default=True): Whether to normalize projection output with L2 norm
        activation (str, default='gelu'): Activation function to use
        enable_mem_efficient (bool, default=False): Whether to use memory efficient attention
        trim (bool, default=True): Whether to trim padded sequences during training
        for_inference (bool, default=False): Whether the model is for inference
        num_classes_cls (int, optional): Number of classes for ONNX export
        use_amp (bool, default=False): Whether to use automatic mixed precision
        return_embed (bool, default=False): Whether to return embeddings
        export_embed (bool, default=False): Whether to export embeddings for ONNX
    """

    def __init__(
        self,
        input_dim,
        proj_dim,
        # network configurations
        pair_input_dim=4,
        pair_extra_dim=0,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        normalize_input: bool = False,
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        num_cls_tokens=1,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        activation="gelu",
        enable_mem_efficient=False,
        head_params=[],
        head_l2_norm=True,
        head_weight_norm=True,
        head_activation="gelu",
        head_compile=False,
        attn_physics_scale=1.0,  # initial scale for physics-based attention
        attn_scale_learnable=False,  # whether to learn the physics scale
        # misc
        trim=True,
        for_inference=False,
        num_classes_cls=None,  # for onnx export
        use_amp=False,
        return_embed=False,
        export_embed=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.num_classes_cls = num_classes_cls
        self.use_amp = use_amp
        self.return_embed = return_embed
        self.export_embed = export_embed

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            add_bias_kv=False,
            activation=activation,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
            enable_mem_efficient=enable_mem_efficient,
        )

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info("cfg_block: %s" % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info("cfg_cls_block: %s" % str(cfg_cls_block))

        self.pair_extra_dim = pair_extra_dim
        self.embed = (
            Embed(input_dim, embed_dims, activation=activation, normalize_input=normalize_input)
            if len(embed_dims) > 0
            else nn.Identity()
        )
        self.pair_embed = (
            PairEmbed(
                pair_input_dim,
                pair_extra_dim,
                pair_embed_dims + [cfg_block["num_heads"]],
                remove_self_pair=remove_self_pair,
                use_pre_activation_pair=use_pre_activation_pair,
                for_onnx=for_inference,
                normalize_input=normalize_input,
            )
            if (
                (pair_embed_dims is not None and len(pair_embed_dims) > 0) 
                and (pair_input_dim + pair_extra_dim > 0)
            )
            else None
        )
        
        if self.pair_embed is None:
            _logger.warning("Pair embedding is not used.")
            self.physics_scale = attn_scale_learnable
        else:
            if attn_scale_learnable:
                self.physics_scale = nn.Parameter(torch.tensor(attn_physics_scale))
            else:
                self.physics_scale = attn_scale_learnable
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList(
            [Block(**cfg_cls_block) for _ in range(num_cls_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # construct final FC layers
        head_params_input = copy.deepcopy(head_params)
        head_params = []
        for fc_param in head_params_input:
            if isinstance(fc_param, int):
                head_params.append((fc_param, 0))
            else:
                head_params.append(fc_param)
        if len(head_params) > 0:
            self.head = DINOHead(
                in_dim=embed_dim,
                proj_dim=proj_dim,
                fc_params=head_params,
                l2_norm=head_l2_norm,
                weight_norm=head_weight_norm,
                activation=head_activation,
            )
            try:
                if head_compile:
                    fc = torch.compile(self.head)
                    self.head = fc
            except Exception as e:
                _logger.error(f"Failed to compile FC: {e}; falling back to standard")
        else:
            self.head = nn.Identity()

        # init
        self.num_cls_tokens = num_cls_tokens
        self.cls_token = nn.Parameter(
            torch.zeros(num_cls_tokens, 1, embed_dim), requires_grad=True
        )
        trunc_normal_(self.cls_token, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def forward(self, particles, mask=None, uu=None, uu_idx=None, use_head: bool = True, **ignore):
        """Forward pass of the Particle Transformer.

        Args:
            particles (torch.Tensor): Particle features tensor of shape (batch_size, num_particles, feature_dim)
                - The first 4 dimensions are assumed to be (px, py, pz, E) for kinematic features.
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, num_particles)
                indicating valid particles (1) vs padding (0)
            uu (torch.Tensor, optional): Pre-computed pairwise features tensor. If uu_idx is None,
                shape is (batch_size, pair_feature_dim, num_particles, num_particles).
                If uu_idx is provided, shape is (batch_size, pair_feature_dim, num_pairs).
            uu_idx (torch.Tensor, optional): Indices tensor of shape (batch_size, 2, num_pairs)
                specifying particle pairs for sparse pairwise features. If provided, uu is interpreted
                as sparse pairwise features.

        Returns:
            torch.Tensor: If self.head is None, returns class token embeddings of shape
                (batch_size, embed_dim). Otherwise, returns classification logits of shape
                (batch_size, proj_dim).
        """
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, num_particles)
        # prepare inputs
        v = particles[..., :4]  # (px, py, pz, E)
        v = v.permute(0, 2, 1)  # (batch_size, 4, num_particles)
        x = particles.permute(0, 2, 1)  # (batch_size, feature_dim, num_particles)
        return self._forward(x=x, v=v, mask=mask, uu=uu, uu_idx=uu_idx, use_head=use_head)

    def _forward(self, x, v=None, mask=None, uu=None, uu_idx=None, use_head: bool = True):
        """(original) Forward pass of the Particle Transformer.

        Args:
            x (torch.Tensor): Particle features tensor of shape (batch_size, feature_dim, num_particles)
            v (torch.Tensor, optional): Particle momentum 4-vectors (px, py, pz, E) of shape
                (batch_size, 4, num_particles). Used for computing kinematic pair features.
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, num_particles)
                indicating valid particles (1) vs padding (0)
            uu (torch.Tensor, optional): Pre-computed pairwise features tensor. If uu_idx is None,
                shape is (batch_size, pair_feature_dim, num_particles, num_particles).
                If uu_idx is provided, shape is (batch_size, pair_feature_dim, num_pairs).
            uu_idx (torch.Tensor, optional): Indices tensor of shape (batch_size, 2, num_pairs)
                specifying particle pairs for sparse pairwise features. If provided, uu is interpreted
                as sparse pairwise features.

        Returns:
            torch.Tensor: If self.head is None, returns class token embeddings of shape
                (batch_size, embed_dim). Otherwise, returns classification logits of shape
                (batch_size, proj_dim).
        """
        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.amp.autocast(enabled=self.use_amp, device_type=x.device.type):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(
                    -1, v.size(-1), v.size(-1)
                )  # (N*num_heads, P, P)
                attn_mask = attn_mask * self.physics_scale

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(
                self.num_cls_tokens, x.size(1), -1
            )  # (N_cls_token, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            cls_tokens = self.norm(cls_tokens)  # (N_cls_token, N, C)
            x_cls = cls_tokens[0]  # (N, C), only use the first cls token

            if use_head:
                # fc
                return self.head(x_cls), x_cls
            else:
                return x_cls

    def freeze_backbone(self):
        """Freeze the backbone (transformer) layers."""
        # freeze all layers
        for param in self.parameters():
            param.requires_grad = False
        # re-enable the last layer
        for param in self.head.parameters():
            param.requires_grad = True

    def reset_head(self):
        """Reset the head (final classification layer) for a new number of classes."""
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
