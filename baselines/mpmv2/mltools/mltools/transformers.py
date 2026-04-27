"""Some classes to describe transformer architectures."""

import logging

import torch as T
import torch.nn.functional as F
from torch import nn

from .attention import flash_cross_attention, flash_self_attention, standard_attention
from .torch_utils import append_dims

log = logging.getLogger(__name__)


def pos_embed(embed_dim: int, max_seq_len: int):
    """Create the positional embedding for the transformer."""
    assert embed_dim % 2 == 0

    # Create the increasing frequencies for the sin and cos functions
    omega = T.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    # Get the positions from the max sequence length
    pos = T.arange(max_seq_len, dtype=float).reshape(-1)

    # Create the matrix using the outer product of the positions and frequencies
    out = omega.unsqueeze(0) * pos.unsqueeze(-1)  # (S, D/2)

    # Embed using sin and cos functions then combine
    emb_sin = T.sin(out)  # (S, D/2)
    emb_cos = T.cos(out)  # (S, D/2)
    pos_emb = T.cat([emb_sin, emb_cos], axis=1)  # (S, D)

    return pos_emb.unsqueeze(0).float()  # For batch dimension


def precompute_freqs_cis(x: T.Tensor, theta: float = 10000.0):
    """Precompute the frequencies for the rotary positional encoding."""
    _B, S, D = x.shape
    t = T.arange(S, device=x.device, dtype=T.float32)
    freqs = 1.0 / (theta ** (T.arange(0, D, 2).float() / D))
    freqs = T.outer(t, freqs)
    return T.polar(T.ones_like(freqs), freqs)


def rope(x: T.Tensor, freqs_cis: T.Tensor) -> T.Tensor:
    """Rotate the input tensor using the precomputed frequencies."""
    B, S, D = x.shape
    q = T.view_as_complex(x.float().reshape(B, S, D // 2, 2))
    q = T.view_as_real(q * freqs_cis)
    return q.view_as(x).type_as(x)


def pack(
    x: T.Tensor,
    mask: T.Tensor | None = None,
    ctxt: T.Tensor | None = None,
) -> T.Tensor:
    """Undo all padding and compress the sequence."""
    if mask is None:
        log.warning("Packing without a mask is not recommended!")
        mask = T.ones(x.shape[:-1], dtype=T.bool, device=x.device)

    # Get the culens and maxlen variables needed by the flash attention func
    seqlens = mask.sum(dim=-1)
    culens = F.pad(T.cumsum(seqlens, dim=-1), (1, 0), value=0).to(T.int32)
    maxlen = seqlens.max().item()

    # Context info gets tricky because it may need to be repeated
    if ctxt is not None:
        if (dim_diff := x.dim() - ctxt.dim()) > 0:  # Expand then pack (no mem copy)
            ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
            ctxt = ctxt.expand(*x.shape[:-1], -1)
        ctxt = ctxt[mask]

    return x[mask], ctxt, culens, maxlen


def unpack(x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
    """Take a compressed sequence and unpack it to a padded tensor."""
    out = T.zeros((*mask.shape, x.shape[-1]), dtype=x.dtype, device=x.device)
    out[mask] = x
    return out


def add_registers(
    x: T.Tensor,
    reg: T.Tensor,
    mask: T.BoolTensor,
    attn_mask: T.BoolTensor | None = None,
    attn_bias: T.Tensor | None = None,
    add_to_both: bool = False,
) -> tuple:
    """Add registers to the front of the input and accomidate the mask.

    add_to_both indicates whether to modify the attn_mask and bias at both the recv
    and send dimenstions. This is primarily because the encoder and decoder use
    these differently. In the encoder the attn mask is between the kv and x tensors
    while in the decoder the attn mask is between the x and x tensors.
    """
    # expand the registers so they can be broadcasted for the whole batch
    reg = reg.expand(x.size(0), -1, x.shape[-1])
    nreg = reg.shape[1]

    # add the registers to the FRONT of the input
    x = T.cat([reg, x], dim=-2)  # Sequence dimension

    # Add the mask for the registers with trues at the front
    if mask is not None:
        mask = F.pad(mask, (nreg, 0), value=True)

    # Add the attention mask for the registers
    # The attention mask is b x recv x send
    # We are adding to the recv dimension ONLY!!!
    if attn_mask is not None:
        attn_mask = F.pad(attn_mask, (nreg * add_to_both, 0, nreg, 0), value=True)

    # Add an attention bias of zero for the registers
    if attn_bias is not None:
        attn_bias = F.pad(attn_bias, (0, 0, nreg * add_to_both, 0, nreg, 0), value=0)

    return x, mask, attn_mask, attn_bias


class RMSNorm(nn.Module):
    """Root Mean Square Normalisation layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(T.ones(dim))
        self.const = dim ** (-0.5)

    def forward(self, x: T.Tensor) -> T.Tensor:
        norm = T.linalg.norm(x.float(), dim=-1, keepdim=True)
        return x * self.scale / (norm * self.const + 1e-8).to(x.dtype)


class QKNorm(nn.Module):
    """Combines the normalisation of the query and key tensors."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)

    def forward(self, q: T.Tensor, k: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        return self.q_norm(q), self.k_norm(k)


class RotaryEmbedding(nn.Module):
    """Applies rotary positional embedding for relative encoding."""

    def __init__(self, dim: int, theta: int = 10_000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("freqs_cis", T.empty(0))

    def _update_freqs_cis(self, x: T.Tensor) -> None:
        """Reset the tables buffer if the sequence / device has changed."""
        new_len = x.shape[1] != self.freqs_cis.shape[0]
        new_device = x.device != self.freqs_cis.device
        if new_len or new_device:
            self.freqs_cis = precompute_freqs_cis(x, self.theta)

    def forward(self, q: T.Tensor, k: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        self._update_freqs_cis(q)
        q = rope(q, self.freqs_cis)
        self._update_freqs_cis(k)
        k = rope(k, self.freqs_cis)
        return q, k


class Residual(nn.Module):
    """Wraps a module with a normalisation layer, residual connection, and gating.

    If context is provided, it is used for adaptive normalisation and gating.
    Gating is always initialised as zero, so the module is initially bypassed.
    """

    def __init__(
        self,
        fn: nn.Module,
        dim: int = 0,
        ctxt_dim: int = 0,
    ) -> None:
        """Parameters
        ----------
        fn : nn.Module
            The module to wrap. Must be non-resizing.
        dim : int
            The dimension of the input and output.
            If zero we will try get it from the fn module.
        ctxt_dim : int, optional
            The dimension of the context, by default 0.
            Used in the modulator to determine the scale, shift and gate.
        """
        super().__init__()
        self.dim = dim or fn.dim
        self.fn = fn
        self.ctxt_dim = ctxt_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        if ctxt_dim:
            self.ctxt_layer = nn.Linear(ctxt_dim, 3 * dim)
        else:
            self.gate = nn.Parameter(T.empty(dim))  # LayerScale with zero init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        if self.ctxt_dim:
            nn.init.constant_(self.ctxt_layer.bias, 0)
            nn.init.uniform_(self.ctxt_layer.weight, 0)
        else:
            nn.init.constant_(self.gate, 0)

    def __repr__(self) -> str:
        return f"Res-{self.fn}"

    def forward(
        self,
        x: T.Tensor,
        *args,
        ctxt: T.Tensor | None = None,
        **kwargs,
    ) -> T.Tensor:
        if self.ctxt_dim:
            ctxt_out = append_dims(self.ctxt_layer(F.silu(ctxt)), x.dim(), dim=1)
            scale, shift, gate = ctxt_out.chunk(3, dim=-1)
            tmp = self.norm(x) * (scale + 1) + shift
        else:
            gate = self.gate
            tmp = self.norm(x)
        return x + self.fn(tmp, *args, **kwargs) * gate


class SwiGLUNet(nn.Module):
    """Simple gated bilinear feedfoward network with the Swish activation."""

    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim  # Usefull for when wrapping the module in residual
        self.lin1 = nn.Linear(dim, 2 * mult * dim)
        self.lin2 = nn.Linear(mult * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(self.drop(F.silu(x1) * x2))


class Attention(nn.Module):
    """Basic multiheaded attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        dropout: float = 0,
        do_rotary: bool = False,
        do_qknorm: bool = False,
    ) -> None:
        """Initialise the attention block.

        Parameters
        ----------
        dim : int
            The dimension of the input and output.
        num_heads : int, optional
            The number of attention heads, by default 1.
        dropout : float, optional
            The dropout probability, by default 0.
        do_rotary : bool, optional
            Whether to use rotary positional encoding, by default False.
        do_qknorm : bool, optional
            Whether to use RMSNorm on the query and key, by default False.
        """
        super().__init__()
        assert dim % num_heads == 0, "Dim must be divisible by the number of heads!"

        # Attributes
        self.dim = dim
        self.num_heads = num_heads
        self.attn_dim = dim // num_heads
        self.dropout = dropout
        self.do_rotary = do_rotary
        self.do_qknorm = do_qknorm

        # Better parallelism for self-attention when using parameters directly
        self.attn_in = nn.Linear(dim, 3 * dim)
        self.attn_out = nn.Linear(dim, dim)

        # Optional extra layers
        self.rotary = RotaryEmbedding(dim) if do_rotary else None
        self.qk_norm = QKNorm(self.attn_dim) if do_qknorm else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the parameters."""
        nn.init.xavier_uniform_(self.attn_in.weight)
        nn.init.xavier_uniform_(self.attn_out.weight)
        nn.init.constant_(self.attn_in.bias, 0)
        nn.init.constant_(self.attn_out.bias, 0)

    def forward(
        self,
        x: T.Tensor,
        mask: T.BoolTensor | None = None,
        kv: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        attn_mask: T.BoolTensor | None = None,
        attn_bias: T.Tensor | None = None,
        culens: T.Tensor | None = None,
        maxlen: int | None = None,
        causal: bool = False,
        kv_culens: T.Tensor | None = None,
        kv_maxlen: int | None = None,
    ) -> T.Tensor:
        """Dispatch to the appropriate attention function based on the inputs."""
        drop = self.dropout if self.training else 0.0

        # If providing the input with culens and maxlen, we assume packed attention
        if culens is not None and maxlen is not None:
            assert attn_mask is None, "Packed attn does not support attention masks!"
            assert attn_bias is None, "Packed attn does not support attention bias!"
            assert not self.do_rotary, "Packed attn does not support rotary emb!"

            if kv is None:
                a_out = flash_self_attention(
                    x,
                    culens,
                    maxlen,
                    drop,
                    causal,
                    self.num_heads,
                    self.attn_in,
                    self.qk_norm,
                )

            else:
                a_out = flash_cross_attention(
                    x,
                    culens,
                    maxlen,
                    kv,
                    kv_culens,
                    kv_maxlen,
                    drop,
                    causal,
                    self.num_heads,
                    self.attn_in,
                    self.qk_norm,
                )

        # Standard attention with masks and biases
        else:
            a_out = standard_attention(
                x,
                kv,
                mask,
                kv_mask,
                attn_mask,
                attn_bias,
                drop,
                causal,
                self.num_heads,
                self.attn_in,
                self.rotary,
                self.qk_norm,
            )

        return self.attn_out(a_out)


class EncoderBlock(nn.Module):
    """Building block for the Transformer Encoder containing MHSA and FFN."""

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        ff_config: dict | None = None,
        attn_config: dict | None = None,
    ) -> None:
        """Initialise the encoder block.

        Parameters
        ----------
        dim : int
            The dimension of of the block
        ctxt_dim : int, optional
            The dimension of the context, by default 0
            Used in the residual modulators to determine the scale, shift and gate.
        ff_config : dict, optional
            The keyword arguments for the feedforward network, by default None
        attn_config : dict, optional
            The keyword arguments for the attention block, by default None
        """
        super().__init__()
        self.dim = dim
        attn_config = attn_config or {}
        ff_config = ff_config or {}

        # Residual blocks
        self.attn = Residual(Attention(dim, **attn_config), dim, ctxt_dim)
        self.ff = Residual(SwiGLUNet(dim, **ff_config), dim, ctxt_dim)

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None, **kwargs) -> T.Tensor:
        x = self.attn(x, ctxt=ctxt, **kwargs)
        return self.ff(x, ctxt=ctxt)


class DecoderBlock(nn.Module):
    """Building block for the Transformer Decoder containing SA+CA+FFN."""

    def __init__(
        self,
        dim: int,
        ctxt_dim: int = 0,
        ff_config: dict | None = None,
        attn_config: dict | None = None,
        ca_first: bool = False,
    ) -> None:
        """Initialise the encoder block.

        Parameters
        ----------
        dim : int
            The dimension of of the block
        ctxt_dim : int, optional
            The dimension of the context, by default 0
            Used in the residual modulators to determine the scale, shift and gate.
        ff_config : dict, optional
            The keyword arguments for the feedforward network, by default None
        attn_config : dict, optional
            The keyword arguments for the attention block, by default None
        ca_first : bool, optional
            Whether to do the cross attention before the self attention, by default
            False
        """
        super().__init__()
        self.dim = dim
        attn_config = attn_config or {}
        ff_config = ff_config or {}
        self.ca_first = ca_first
        self.sa = Residual(Attention(dim, **attn_config), dim, ctxt_dim)
        self.ca = Residual(Attention(dim, **attn_config), dim, ctxt_dim)
        self.ff = Residual(SwiGLUNet(dim, **ff_config), dim, ctxt_dim)

    def forward(
        self,
        x: T.Tensor,
        *,  # Indicates that kv is a required but keyword argument
        kv: T.Tensor,
        mask: T.BoolTensor | None = None,
        ctxt: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        attn_mask: T.Tensor | None = None,
        attn_bias: T.Tensor | None = None,
        **kwargs,
    ) -> T.Tensor:
        """Pass through the decoder block."""
        if self.ca_first:
            x = self.ca(x, mask, kv, kv_mask, None, None, ctxt=ctxt, **kwargs)
            x = self.sa(x, mask, None, None, attn_mask, attn_bias, ctxt=ctxt, **kwargs)
        else:
            x = self.sa(x, mask, None, None, attn_mask, attn_bias, ctxt=ctxt, **kwargs)
            x = self.ca(x, mask, kv, kv_mask, None, None, ctxt=ctxt, **kwargs)
        return self.ff(x, ctxt=ctxt)


class Transformer(nn.Module):
    """Simple transformer stack of encoder or decoder blocks.

    Includes option to add registers from: doi.org/10.48550/arXiv.2309.16588.
    Single register can be thought of as the class token.
    """

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        num_layers: int = 6,
        max_seq_len: int = 0,
        num_registers: int = 0,
        do_input_linear: bool = False,
        do_output_linear: bool = False,
        do_absolute_enc: bool = False,
        do_final_norm: bool = False,
        layer_config: dict | None = None,
        inpt_dim: None | int = None,
        outp_dim: None | int = None,
        use_decoder: bool = False,
        pack_inputs: bool = False,
        unpack_output: bool = True,
    ) -> None:
        """Parameters
        ----------
        dim : int, optional
            The dimension of the model, by default 128.
        ctxt_dim : int, optional
            The dimension of the context, by default 0.
        num_layers : int, optional
            The number of layers in the transformer, by default 6.
        max_seq_len : int, optional
            The maximum sequence length, by default 0.
            Needed for absolute positional encoding.
        num_registers : int, optional
            The number of registers to add to the input, by default 0.
        do_input_linear : bool, optional
            Whether to add an input linear layer, by default False.
            Will decouple the input dimension from the transformer dimension.
        do_output_linear : bool, optional
            Whether to add an output linear layer, by default False.
            Will decouple the output dimension from the transformer dimension.
        do_absolute_enc : bool, optional
            Whether to add absolute encoding, by default False.
            Must provide max_seq_len if True.
        do_final_norm : bool, optional
            Whether to add a final layer norm, by default False.
        inpt_dim : None | int, optional
            The input dimension, by default None.
            If None, then will be set to dim.
        outp_dim : None | int, optional
            The output dimension, by default None.
            If None, then will be set to dim.
        use_decoder : bool, optional
            Whether to use the decoder blocks, by default False.
        pack_inputs : bool, optional
            Whether to pack the inputs for optimised training on varlen sequences.
        unpack_output : bool, optional
            If using packed inputs, whether to unpack all outputs, by default True.
        layer_config : dict | None, optional
            The configuration for the encoder/decoder blocks, by default None.
        """
        super().__init__()
        assert not (do_absolute_enc and max_seq_len == 0), "Define max_seq_len!"
        layer_config = layer_config or {}

        self.dim = dim
        self.ctxt_dim = ctxt_dim
        self.num_registers = num_registers
        self.num_layers = num_layers
        self.do_final_norm = do_final_norm
        self.do_input_linear = do_input_linear
        self.do_output_linear = do_output_linear
        self.do_absolute_enc = do_absolute_enc
        self.layer_config = layer_config
        self.inpt_dim = inpt_dim if do_input_linear else dim
        self.outp_dim = outp_dim if do_output_linear else dim
        self.use_decoder = use_decoder
        self.pack_inputs = pack_inputs
        self.unpack_output = unpack_output

        # Base repeated transformer layers
        lyr = DecoderBlock if use_decoder else EncoderBlock
        self.layers = nn.ModuleList([
            lyr(dim, ctxt_dim, **layer_config) for _ in range(num_layers)
        ])

        # Optional layers and features
        if self.do_input_linear:
            self.linear_embed = nn.Linear(inpt_dim, dim)
        if self.do_final_norm:
            self.final_norm = nn.LayerNorm(dim)
        if self.do_output_linear:
            self.linear_out = nn.Linear(dim, outp_dim)
        if self.num_registers:
            self.registers = nn.Parameter(T.randn((1, self.num_registers, dim)) * 1e-3)
        if self.do_absolute_enc:
            self.abs_enc = nn.Parameter(pos_embed(dim, max_seq_len, num_registers))

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Project and encode.

        Why are these seperate?
        - Added flexibility for doing something to the inputs (replacing with null)
          once they are projected into the transformer dimension.
        """
        return self.encode(self.project(x), **kwargs)

    def project(self, x: T.Tensor) -> T.Tensor:
        """Project the input to the transformer dimension and add absolute encoding."""
        if self.do_input_linear:
            x = self.linear_embed(x)
        if self.do_absolute_enc:
            x = x + self.abs_enc[:, : x.shape[-2], :]  # Trims to the sequence length
        return x

    def encode(
        self,
        x: T.Tensor,
        mask: T.BoolTensor | None = None,
        ctxt: T.Tensor | None = None,
        attn_mask: T.BoolTensor | None = None,
        attn_bias: T.Tensor | None = None,
        kv: T.Tensor | None = None,
        kv_mask: T.BoolTensor | None = None,
        **kwargs,
    ) -> T.Tensor:
        """Pass through all layers of the transformer."""
        assert not (
            self.num_registers and "culens" in kwargs
        ), "Cannot add registers to inputs which are already packed!"
        if self.num_registers:
            x, mask, attn_mask, attn_bias = add_registers(
                x,
                self.registers,
                mask,
                attn_mask,
                attn_bias,
                add_to_both=self.use_decoder,  # Changes the attn mask for the decoder
            )
        if self.pack_inputs and "culens" not in kwargs:
            x, ctxt, culens, maxlen = pack(x, mask, ctxt)
            kwargs["culens"] = culens  # Add to the kwargs for the forward pass
            kwargs["maxlen"] = maxlen
            if kv is not None and "kv_culens" not in kwargs:
                kv, _, kv_culens, kv_maxlen = pack(kv, kv_mask, None)
                kwargs["kv_culens"] = kv_culens
                kwargs["kv_maxlen"] = kv_maxlen
        for layer in self.layers:
            x = layer(
                x,
                mask=mask,
                ctxt=ctxt,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
                kv=kv,
                kv_mask=kv_mask,
                **kwargs,
            )
        if self.do_final_norm:
            x = self.final_norm(x)
        if self.do_output_linear:
            x = self.linear_out(x)
        if "culens" in kwargs:
            if self.unpack_output:
                return unpack(x, mask)
            return x, culens, maxlen
        return x

    def remove_registers(self, x: T.Tensor) -> T.Tensor:
        """Remove the registers from the front of the input."""
        return x[:, : self.num_registers], x[:, self.num_registers :]

    def get_combined_mask(self, mask: T.BoolTensor | None) -> T.BoolTensor | None:
        """Get a mask which can be used for the combined register+sequence tensor."""
        if self.num_registers == 0:
            return mask
        if mask is None:
            return None
        return F.pad(mask, (self.num_registers, 0), value=True)


class CrossAttentionEncoder(nn.Module):
    """Lopsided transformer which upades two point clouds x1, x2.

    Only applied self-attention to the second point cloud.
    """

    def __init__(
        self,
        *,
        dim: int = 128,
        x1_input_dim: int = 0,
        x2_input_dim: int = 0,
        x1_output_dim: int = 0,
        x2_output_dim: int = 0,
        ctxt_dim: int = 0,
        num_layers: int = 2,
        x1_max_seq_len: int = 0,
        x2_max_seq_len: int = 0,
        x1_absolute_enc: bool = False,
        x2_absolute_enc: bool = False,
        enc_config: dict | None = None,
        dec_config: dict | None = None,
        use_decoder: bool = False,
        pack_inputs: bool = False,
        unpack_output: bool = True,
    ) -> None:
        super().__init__()
        assert not (x1_absolute_enc and x1_max_seq_len == 0), "Define x1 max_seq_len!"
        assert not (x2_absolute_enc and x2_max_seq_len == 0), "Define x2 max_seq_len!"
        enc_config = enc_config or {}
        dec_config = dec_config or {}
        self.dim = dim
        self.x1_input_dim = x1_input_dim or dim
        self.x2_input_dim = x2_input_dim or dim
        self.x1_output_dim = x1_output_dim or dim
        self.x2_output_dim = x2_output_dim or dim
        self.ctxt_dim = ctxt_dim
        self.num_layers = num_layers
        self.x1_max_seq_len = x1_max_seq_len
        self.x2_max_seq_len = x2_max_seq_len
        self.x1_absolute_enc = x1_absolute_enc
        self.x2_absolute_enc = x2_absolute_enc
        self.use_decoder = use_decoder
        self.pack_inputs = pack_inputs
        self.unpack_output = unpack_output

        # Each layer needs a decoder block and an encoder block
        self.dec_layers = nn.ModuleList([
            DecoderBlock(dim, ctxt_dim, **dec_config) for _ in range(num_layers)
        ])
        self.enc_layers = nn.ModuleList([
            EncoderBlock(dim, ctxt_dim, **enc_config) for _ in range(num_layers)
        ])

        # Input, output and final norm layers, always present
        self.x1_linear_in = nn.Linear(self.x1_input_dim, dim)
        self.x2_linear_in = nn.Linear(self.x2_input_dim, dim)
        self.x1_linear_out = nn.Linear(dim, self.x1_output_dim)
        self.x2_linear_out = nn.Linear(dim, self.x2_output_dim)
        self.x1_final_norm = nn.LayerNorm(dim)
        self.x2_final_norm = nn.LayerNorm(dim)

        # Absolute encoding for the two point clouds is optional
        if x1_absolute_enc:
            self.x1_abs_enc = nn.Parameter(pos_embed(dim, x1_max_seq_len))
        if x2_absolute_enc:
            self.x2_abs_enc = nn.Parameter(pos_embed(dim, x2_max_seq_len))

    def forward(
        self,
        x1: T.Tensor,
        x2: T.Tensor,
        x1_mask: T.BoolTensor | None = None,
        x2_mask: T.BoolTensor | None = None,
        ctxt: T.Tensor | None = None,
        x2_attn_mask: T.BoolTensor | None = None,
        x2_attn_bias: T.Tensor | None = None,
        x2_causal: bool = False,
    ) -> T.Tensor:
        """Pass through all layers of the transformer."""
        x1 = self.x1_linear_in(x1)
        x2 = self.x2_linear_in(x2)
        if self.x1_absolute_enc:
            x1 = x1 + self.x1_abs_enc[:, : x1.shape[-2], :]
        if self.x2_absolute_enc:
            x2 = x2 + self.x2_abs_enc[:, : x2.shape[-2], :]
        if self.pack_inputs:
            x1, x1_ctxt, x1_culens, x1_maxlen = pack(x1, x1_mask, ctxt)
            x2, x2_ctxt, x2_culens, x2_maxlen = pack(x2, x2_mask, ctxt)
        else:
            x1_ctxt, x2_ctxt = ctxt, ctxt
            x1_culens, x2_culens = None, None
            x1_maxlen, x2_maxlen = None, None

        for dec_layer, enc_layer in zip(self.dec_layers, self.enc_layers, strict=True):
            x2 = dec_layer(
                x=x2,
                kv=x1,
                ctxt=x2_ctxt,
                mask=x2_mask,
                kv_mask=x1_mask,
                attn_mask=x2_attn_mask,
                attn_bias=x2_attn_bias,
                culens=x2_culens,
                maxlen=x2_maxlen,
                kv_culens=x1_culens,
                kv_maxlen=x1_maxlen,
                causal=x2_causal,
            )
            x1 = enc_layer(
                x=x1,
                kv=x2,
                ctxt=x1_ctxt,
                mask=x1_mask,
                kv_mask=x2_mask,
                culens=x1_culens,
                maxlen=x1_maxlen,
                kv_culens=x2_culens,
                kv_maxlen=x2_maxlen,
            )
        x1 = self.x1_final_norm(x1)
        x2 = self.x2_final_norm(x2)
        x1 = self.x1_linear_out(x1)
        x2 = self.x2_linear_out(x2)
        if self.pack_inputs:
            if self.unpack_output:
                return unpack(x1, x1_mask), unpack(x2, x2_mask)
            return (x1, x1_culens, x1_maxlen), (x2, x2_culens, x2_maxlen)
        return x1, x2


class ClassAttentionPooling(nn.Module):
    """Pooling operation that uses attention."""

    def __init__(
        self,
        *,
        dim: int = 128,
        ctxt_dim: int = 0,
        num_layers: int = 1,
        outp_dim: int = 0,
        inpt_dim: int = 0,
        layer_config: dict | None = None,
    ) -> None:
        """Parameters
        ----------
        dim : int, optional
            The dimension of the input and output embeddings. Default is 128.
        ctxt_dim : int, optional
            The dimension of the context embeddings. Default is 0.
        num_layers : int, optional
            The number of cross attention pooling layers. Default is 1.
        outp_dim : int, optional
            The dimension of the output embeddings. If not provided, defaults to `dim`.
        inpt_dim : int, optional
            The dimension of the input embeddings. If not provided, defaults to `dim`.
        layer_config : dict or None, optional
            Additional configuration for the layers. Default is None.
        """
        super().__init__()
        layer_config = layer_config or {}
        self.dim = dim
        self.ctxt_dim = ctxt_dim
        self.num_layers = num_layers
        self.outp_dim = outp_dim or dim
        self.inpt_dim = inpt_dim or dim

        # The single learnable global token
        self.global_token = nn.Parameter(T.randn((1, 1, self.dim)))

        # Main cross attention layers
        self.layers = nn.ModuleList([
            EncoderBlock(self.dim, ctxt_dim, **layer_config) for _ in range(num_layers)
        ])

        # Extra layers
        self.linear_in = nn.Linear(self.inpt_dim, dim)
        self.final_norm = nn.LayerNorm(dim)
        self.linear_out = nn.Linear(dim, self.outp_dim)

    def forward(
        self, x: T.Tensor, mask: T.BoolTensor | None = None, **kwargs
    ) -> T.Tensor:
        """Perform class attention pooling on a sequence."""
        x = self.linear_in(x)

        # If x is packed, then so too must be the global token
        if "culens" in kwargs:
            culens = kwargs["culens"]
            maxlen = kwargs["maxlen"]
            B = culens.size(0) - 1
            g = self.global_token.squeeze(1).expand(B, self.dim)
            kwargs["culens"] = T.arange(B + 1, device=culens.device, dtype=culens.dtype)
            kwargs["maxlen"] = 1
            kwargs["kv_culens"] = culens
            kwargs["kv_maxlen"] = maxlen

        # Otherwise we broadcast the global token to match the batch size
        else:
            g = self.global_token.expand(x.shape[0], 1, self.dim)

        # Pass through the layers
        for layer in self.layers:
            g = layer(g, kv=x, kv_mask=mask, **kwargs)
        g = self.final_norm(g)
        g = self.linear_out(g)

        # If not packed, then we pop out the sequence dimension
        # If packed, then the format is already correct
        if "culens" not in kwargs:
            g.squeeze_(-2)
        return g
