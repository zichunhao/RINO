"""Code for everything convolutional."""

import logging

import numpy as np
import torch as T
from torch import nn
from torch.nn.functional import group_norm, interpolate, scaled_dot_product_attention

from .mlp import MLP
from .torch_utils import append_dims, zero_module

log = logging.getLogger(__name__)


def conv_nd(dims, *args, **kwargs) -> nn.Module:
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def drop_nd(dims, *args, **kwargs) -> nn.Module:
    """Create a 1D, 2D, or 3D droupout module."""
    if dims == 1:
        return nn.Dropout(*args, **kwargs)
    if dims == 2:
        return nn.Dropout2d(*args, **kwargs)
    if dims == 3:
        return nn.Dropout3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs) -> nn.Module:
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class ConditionedModule(nn.Module):
    """Base class for models that need context when processing data."""


class ConditionedSequential(nn.Sequential):
    """Sequential model that can provide context to each layers if required."""

    def forward(self, inpt, ctxt):
        """Pass through all submodules."""
        for module in self:
            if isinstance(module, ConditionedModule):
                inpt = module(inpt, ctxt)
            else:
                inpt = module(inpt)
        return inpt


class AdaGN(ConditionedModule):
    """A module that implements an adaptive group normalization layer."""

    def __init__(self, ctxt_dim: int, c_out: int, nrm_groups: int, eps=1e-5) -> None:
        """Parameters
        ----------
        ctxt_dim : int
            The dimension of the context tensor.
        c_out : int
            The number of output channels.
        nrm_groups : int
            The number of groups for group normalization.
        eps : float, optional
            A small value added to the denominator for numerical stability.
            Default: 1e-5.
        """
        super().__init__()
        self.ctxt_dim = ctxt_dim
        self.c_out = c_out
        self.num_groups = nrm_groups
        self.eps = eps
        self.layer = nn.Linear(ctxt_dim, c_out * 2)

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor) -> T.Tensor:
        """Apply conditioning to inputs."""
        scale, shift = self.layer(ctxt).chunk(2, dim=-1)
        scale = append_dims(scale, inpt.ndim) + 1  # + 1 to not kill on init
        shift = append_dims(shift, inpt.ndim)
        inpt = group_norm(inpt, self.num_groups, eps=self.eps)
        return T.addcmul(shift, inpt, scale)

    def __str__(self) -> str:
        return f"AdaGN({self.ctxt_dim}, {self.c_out})"

    def __repr__(self) -> str:
        return str(self)


class ResNetBlock(ConditionedModule):
    """A residual convolutional block.

    - Can change channel dimensions but not spacial.
    - All convolutions are stride 1 with padding 1.
    - May also take in some context tensor which is injected using AdaGN.
    - Forward pass applies the following:
        - AdaGN->Act->Conv->AdaGN->Act->Drop->0Conv + skip_connection
    """

    def __init__(
        self,
        inpt_channels: int,
        ctxt_dim: int = 0,
        outp_channels: int | None = None,
        kernel_size: int = 3,
        dims: int = 2,
        act: str = "SiLU",
        drp: float = 0,
        nrm_groups: int = 1,
    ) -> None:
        """Parameters
        ----------
        inpt_channels : int
            The number of input channels.
        ctxt_dim : int, optional
            The dimension of the context tensor. Default: 0.
        outp_channels : int, optional
            The number of output channels. Default: None.
        kernel_size : int, optional
            The size of the convolution kernel. Default: 3.
        dims : int, optional
            The number of dimensions (2 is an image). Default: 2.
        act : str, optional
            The activation function to use. Default: "lrlu".
        drp : float, optional
            The dropout rate. Default: 0.
        nrm_groups : int, optional
            The number of groups for group normalization. Default: 1.
        """
        super().__init__()

        # Attributes
        self.inpt_channels = inpt_channels
        self.outp_channels = outp_channels or inpt_channels
        self.ctxt_dim = ctxt_dim

        # The method for normalisation is where the context is injected
        def get_norm(c_out) -> AdaGN | nn.GroupNorm:
            if ctxt_dim:
                return AdaGN(ctxt_dim, c_out, nrm_groups)
            return nn.GroupNorm(nrm_groups, c_out)

        # Create the main layer structure of the network
        self.layers = ConditionedSequential(
            get_norm(inpt_channels),
            getattr(nn, act)(),
            conv_nd(dims, inpt_channels, outp_channels, kernel_size, padding=1),
            get_norm(outp_channels),
            getattr(nn, act)(),
            drop_nd(dims, drp),
            zero_module(
                conv_nd(dims, outp_channels, outp_channels, kernel_size, padding=1)
            ),
        )

        # Create the skip connection, using a 1x1 conv to change channel size
        if self.inpt_channels == self.outp_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, inpt_channels, outp_channels, 1)

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        """Pass through the layers with the residual connection."""
        return self.layers(inpt, ctxt) + self.skip_connection(inpt)

    def __str__(self) -> str:
        string = f"ResNetBlock({self.inpt_channels}, {self.outp_channels}"
        if self.ctxt_dim:
            string += f", ctxt={self.ctxt_dim}"
        string += ")"
        return string

    def __repr__(self) -> str:
        return str(self)


class MultiHeadedAttentionBlock(ConditionedModule):
    """A self attention block specifically designed for images.

    This layer essentailly flattens the image's spacial dimensions, making it a sequence
    where the length equals the original resolution. The dimension of each element of
    the sequence is the number of each channels.

    Then the message passing occurs, which is permutation invariant, using the exact
    same operations as a standard transformer except we use 1x1 convolutions instead of
    linear projections (same maths, but optimised performance)
    """

    def __init__(
        self,
        inpt_channels: int,
        inpt_shape: tuple,
        ctxt_dim: int = 0,
        num_heads: int = 1,
        nrm_groups: int = 1,
        do_pos_encoding: bool = True,
    ) -> None:
        super().__init__()

        # Ensure that the number of channels is divisible by the number of heads
        assert inpt_channels % num_heads == 0

        # Class attributes
        self.inpt_channels = inpt_channels
        self.inpt_shape = inpt_shape
        self.ctxt_dim = ctxt_dim
        self.num_heads = num_heads
        self.attn_dim = inpt_channels // num_heads
        self.do_pos_encoding = do_pos_encoding

        # The learnable positional encodings
        if self.do_pos_encoding:
            self.pos_enc = nn.Parameter(T.randn(1, inpt_channels, *inpt_shape) * 1e-3)

        # Context is injected via a linear mixing layer
        if ctxt_dim:
            self.ctxt_layer = conv_nd(
                len(inpt_shape),
                inpt_channels + ctxt_dim,
                inpt_channels,
                1,
            )

        # The layers used in the attention operation
        self.norm = nn.GroupNorm(nrm_groups, inpt_channels)
        self.qkv = conv_nd(len(inpt_shape), inpt_channels, inpt_channels * 3, 1)
        self.out_conv = zero_module(
            conv_nd(len(inpt_shape), inpt_channels, inpt_channels, 1)
        )

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        """Apply the model the message passing, context tensor is not used."""
        # Save important shapes for permuting the attention heads
        b, c, *spatial = inpt.shape

        # Make a copy of the input, double checked that this does copy!
        qkv = inpt

        # Normalise
        qkv = self.norm(qkv)

        # Add the positional encoding
        if self.do_pos_encoding:
            qkv = qkv + self.pos_enc

        # Mix in the context tensor
        if self.ctxt_dim:
            ctxt = append_dims(ctxt, inpt.ndim).expand(-1, -1, *spatial)
            qkv = self.ctxt_layer(T.cat([qkv, ctxt], 1))

        # Perform the projections
        qkv = self.qkv(qkv)

        # Flatten out the spacial dimensions them swap to get: B, 3N, HxW, h_dim
        qkv = qkv.view(b, self.num_heads * 3, self.attn_dim, -1).transpose(-1, -2)
        q, k, v = T.chunk(qkv.contiguous(), 3, dim=1)

        # Perform standard scaled dot product attention
        a_out = scaled_dot_product_attention(q, k, v)

        # Concatenate the heads together to get back to: B, c, H, W
        a_out = a_out.transpose(-1, -2).contiguous().view(b, c, *spatial)

        # Apply redidual update
        return inpt + self.out_conv(a_out)


class ConvNet(nn.Module):
    """A very simple convolutional neural network which halves the spacial dimension
    with each block while doubling the number of channels.

    - Attention operations occur after a certain number of downsampling steps
    - Downsampling is performed using 2x2 average pooling
    - Ends with an MLP
    """

    def __init__(
        self,
        inpt_size: list,
        inpt_channels: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        num_blocks_per_layer: int = 1,
        attn_resolution: int = 8,
        start_channels: int = 32,
        channel_mult: list | None = None,
        resnet_config: dict | None = None,
        attn_config: dict | None = None,
        mlp_config: dict | None = None,
    ) -> None:
        """Initialize the network.

        Parameters
        ----------
        inpt_size : list
            The size of the input image. Can be 1D, 2D, or 3D.
        inpt_channels : int
            The number of channels in the input image.
        outp_dim : int
            The dimension of the output vector.
        ctxt_dim : int, optional
            The dimension of the context input. Default is 0.
        num_blocks_per_layer : int, optional
            The number of ResNet blocks per layer. Default is 1.
        attn_resolution : int, optional
            The maximum size of spacial dimensions for attention operations.
            Default is 8.
        start_channels : int, optional
            The number of channels at the start of the network. Default is 32.
        channel_mult : list, optional
            The multiplier for the number of channels at each level.
            Also determines the number of levels.
            Default is [1, 2].
        resnet_config : dict, optional
            Configuration for ResNet blocks. Default is None.
        attn_config : dict, optional
            Configuration for attention blocks. Default is None.
        mlp_config : dict, optional
            Configuration for the final dense network. Default is None.
        """
        super().__init__()

        # Safe defaults
        resnet_config = resnet_config or {}
        attn_config = attn_config or {}
        mlp_config = mlp_config or {}
        channel_mult = channel_mult or [1, 2]

        # Class attributes
        self.inpt_size = np.array(inpt_size)
        self.inpt_channels = inpt_channels
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.num_levels = len(channel_mult)

        # The downsampling layer (not learnable)
        dims = len(inpt_size)
        stride = 2 if dims != 3 else (2, 2, 2)
        self.down_sample = avg_pool_nd(dims, kernel_size=stride, stride=stride)

        # The first conv layer sets up the starting channel size
        self.first_block = nn.Sequential(
            conv_nd(dims, inpt_channels, start_channels, 1), nn.SiLU()
        )

        # Work out how many levels there will be to the unet, their sizes and channels
        lvl_dims = [self.inpt_size // (2**i) for i in range(self.num_levels)]
        lvl_ch = [start_channels * i for i in channel_mult]
        current_ch = start_channels

        # Check if the setup results in too small of an image
        if min(lvl_dims[-1]) < 1:
            raise ValueError("Middle shape of UNet is less than 1!")

        # The encoder blocks, build from top to bottom
        self.encoder_blocks = nn.ModuleList()
        for i in range(self.num_levels):
            level_layers = nn.ModuleList()
            for _ in range(num_blocks_per_layer):
                level_layers.append(
                    ResNetBlock(
                        inpt_channels=current_ch,
                        outp_channels=lvl_ch[i],
                        ctxt_dim=self.ctxt_dim,
                        **resnet_config,
                    )
                )
                current_ch = lvl_ch[i]
            if max(lvl_dims[i]) <= attn_resolution:
                level_layers.append(
                    MultiHeadedAttentionBlock(
                        inpt_channels=current_ch,
                        inpt_shape=lvl_dims[i],
                        ctxt_dim=self.ctxt_dim,
                        **attn_config,
                    )
                )

            # Add the level's layers to the block list
            self.encoder_blocks.append(level_layers)

        # Create the dense network
        self.dense = MLP(
            inpt_dim=np.prod(lvl_dims[-1]) * current_ch,  # Final size and channels
            outp_dim=outp_dim,
            ctxt_dim=ctxt_dim,
            **mlp_config,
        )

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None):
        """Forward pass of the network."""
        # Pass through the first convolution layer to embed the channel dimension
        x = self.first_block(x)

        # Pass through the ResNetBlocks and the downsampling
        for i, level in enumerate(self.encoder_blocks):
            for layer in level:
                x = layer(x, ctxt)
            if i < self.num_levels - 1:  # Don't downsample the last level
                x = self.down_sample(x)

        # Flatten and pass through final dense network and return
        x = T.flatten(x, start_dim=1)

        return self.dense(x, ctxt)


class UNet(nn.Module):
    """Image to Image mapping network.

    - Attention operations occur at a specified resolution
    - Downsampling is performed using 2x2 average pooling
    - Upsampling is performed using nearest neighbour
    """

    def __init__(
        self,
        inpt_size: list,
        inpt_channels: int,
        outp_channels: int | None = None,
        ctxt_dim: int = 0,
        ctxt_img_channels: int = 0,
        num_blocks_per_layer: int = 1,
        attn_resolution: int = 8,
        start_channels: int = 32,
        channel_mult: list | None = None,
        zero_out: bool = False,
        resnet_config: dict | None = None,
        attn_config: dict | None = None,
    ) -> None:
        """Parameters
        ----------
        inpt_size : list
            The spacial dimensions of the input image. Can be 1D, 2D, or 3D.
        inpt_channels : int
            The number of channels in the input image.
        outp_channels : int, optional
            The number of channels in the output image. Default: None.
            If None, it will match the input channels.
        ctxt_dim : int, optional
            The dimension of the context tensor. Default: 0.
        ctxt_img_channels : int, optional
            The number of channels in the context image. Default: 0.
        num_blocks_per_layer : int, optional
            The number of ResNet blocks per layer. Default: 1.
        attn_resolution : int, optional
            The spacial resolution at which to start using attention. Default: 8.
        start_channels : int, optional
            The number of channels at the start of the network. Default: 32.
        channel_mult : list, optional
            The multiplier for the number of channels at each level.
            Also determines the number of levels. Default: None.
        zero_out : bool, optional
            Whether to zero out the final convolution layer. Default: False.
        resnet_config : dict, optional
            Configuration for ResNet blocks. Default: None.
        attn_config : dict, optional
            Configuration for attention blocks. Default: None.
        """
        super().__init__()

        # Safe dict defaults
        resnet_config = resnet_config or {}
        attn_config = attn_config or {}
        channel_mult = channel_mult or []

        # Class attributes
        self.inpt_size = np.array(inpt_size)
        self.inpt_channels = inpt_channels
        self.outp_channels = outp_channels or inpt_channels
        self.ctxt_dim = ctxt_dim
        self.ctxt_img_channels = ctxt_img_channels
        self.num_levels = len(channel_mult)

        # The downsampling layer and upscaling layers (not learnable)
        dims = len(inpt_size)
        stride = 2 if dims != 3 else (2, 2, 2)
        self.down_sample = avg_pool_nd(dims, kernel_size=stride, stride=stride)
        self.up_sample = nn.Upsample(scale_factor=2)

        # The first and last conv layer sets up the starting channel size
        self.first_block = nn.Sequential(
            conv_nd(dims, self.inpt_channels + ctxt_img_channels, start_channels, 1),
            nn.SiLU(),
        )
        self.last_block = nn.Sequential(
            nn.SiLU(),
            conv_nd(dims, start_channels, self.outp_channels, 1),
        )
        if zero_out:
            self.last_block = zero_module(self.last_block)

        # Work out how many levels there will be to the unet, their sizes and channels
        lvl_dims = [self.inpt_size // (2**i) for i in range(self.num_levels)]
        lvl_ch = [start_channels * i for i in channel_mult]
        current_ch = start_channels

        # Check if the setup results in too small of an image
        if min(lvl_dims[-1]) < 1:
            raise ValueError("Middle shape of UNet is less than 1!")

        # The encoder blocks, build from top to bottom
        self.encoder_blocks = nn.ModuleList()
        for i in range(self.num_levels - 1):  # Final level are middle blocks
            level_layers = nn.ModuleList()
            for _ in range(num_blocks_per_layer):
                level_layers.append(
                    ResNetBlock(
                        inpt_channels=current_ch,
                        outp_channels=lvl_ch[i],
                        ctxt_dim=self.ctxt_dim,
                        **resnet_config,
                    )
                )
                current_ch = lvl_ch[i]
            if max(lvl_dims[i]) <= attn_resolution:
                level_layers.append(
                    MultiHeadedAttentionBlock(
                        inpt_channels=current_ch,
                        inpt_shape=lvl_dims[i],
                        ctxt_dim=self.ctxt_dim,
                        **attn_config,
                    )
                )

            # Add the level's layers to the block list
            self.encoder_blocks.append(level_layers)

        # The middle part of the UNet at the lowest level
        self.middle_blocks = nn.ModuleList([
            ResNetBlock(
                inpt_channels=lvl_ch[-2],
                outp_channels=lvl_ch[-1],
                ctxt_dim=self.ctxt_dim,
                **resnet_config,
            ),
            ResNetBlock(
                inpt_channels=lvl_ch[-1],
                outp_channels=lvl_ch[-2],
                ctxt_dim=self.ctxt_dim,
                **resnet_config,
            ),
        ])

        # Attention in the middle if the size is small enough
        if max(lvl_dims[-1]) <= attn_resolution:
            self.middle_blocks.insert(
                1,
                MultiHeadedAttentionBlock(
                    inpt_channels=lvl_ch[-1],
                    inpt_shape=lvl_dims[-1],
                    ctxt_dim=self.ctxt_dim,
                    **attn_config,
                ),
            )

        # The decoder layers, a mirror of the encoder
        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(self.num_levels - 1)):  # Final level are middle blocks
            level_layers = nn.ModuleList()
            for j in range(num_blocks_per_layer):
                level_layers.append(
                    ResNetBlock(  # Include special case for concat skip connections
                        inpt_channels=current_ch + lvl_ch[i] * (j == 0),
                        outp_channels=lvl_ch[i],
                        ctxt_dim=self.ctxt_dim,
                        **resnet_config,
                    )
                )
                current_ch = lvl_ch[i]
            if max(lvl_dims[i]) <= attn_resolution:
                level_layers.append(
                    MultiHeadedAttentionBlock(
                        inpt_channels=current_ch,
                        inpt_shape=lvl_dims[i],
                        ctxt_dim=self.ctxt_dim,
                        **attn_config,
                    )
                )

            # Add the level's layers to the block list
            self.decoder_blocks.append(level_layers)

    def forward(
        self,
        x: T.Tensor,
        ctxt: T.Tensor | None = None,
        ctxt_img: T.Tensor | None = None,
    ) -> T.Tensor:
        """Forward pass of the network."""
        # Make sure the input size is expected
        if x.shape[-1] != self.inpt_size[-1]:
            log.warning("Input image does not match the training sample!")

        # Combine the input with the context image
        if self.ctxt_img_channels:
            if ctxt_img.shape != x.shape:
                ctxt_img = interpolate(ctxt_img, x.shape[-2:])
            x = T.cat([x, ctxt_img], 1)

        # Make sure the dtype of the context matches the image
        ctxt = ctxt.type(x.dtype)

        # Pass through the first convolution layer to embed the channel dimension
        x = self.first_block(x)

        # Pass through the encoder
        enc_outs = []
        for level in self.encoder_blocks:
            for layer in level:
                x = layer(x, ctxt)
            enc_outs.append(x)  # Save the output to the buffer
            x = self.down_sample(x)  # Apply the downsampling

        # Pass through the middle blocks
        for block in self.middle_blocks:
            x = block(x, ctxt)

        # Pass through the decoder blocks
        for level in self.decoder_blocks:
            x = self.up_sample(x)  # Apply the upsampling
            x = T.cat([x, enc_outs.pop()], dim=1)  # Concat with buffer
            for layer in level:
                x = layer(x, ctxt)

        # Pass through the final layer
        return self.last_block(x)
