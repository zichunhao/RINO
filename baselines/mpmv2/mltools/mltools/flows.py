"""Functions and classes used to define invertible transformations."""

from collections.abc import Callable
from typing import Any, Literal

import normflows as nf
import numpy as np
import torch as T
from normflows.distributions.base import BaseDistribution, DiagGaussian
from normflows.flows.neural_spline.coupling import PiecewiseRationalQuadraticCoupling
from normflows.utils.masks import create_alternating_binary_mask
from normflows.utils.splines import DEFAULT_MIN_DERIVATIVE
from torch import nn

from .mlp import MLP
from .torch_utils import base_modules


class Uniform(BaseDistribution):
    """Multivariate uniform distribution.

    Needed because default normflows doesnt save low and high as buffers, leading to
    mistakes when using a GPU.
    """

    def __init__(self, shape, low=-1.0, high=1.0):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.d = np.prod(shape)
        self.register_buffer("low", T.tensor(low))
        self.register_buffer("high", T.tensor(high))
        self.log_prob_val = -self.d * np.log(self.high - self.low)

    def forward(self, num_samples=1, context=None) -> tuple:  # noqa: ARG002
        eps = T.rand(
            (num_samples, *self.shape),
            dtype=self.low.dtype,
            device=self.low.device,
        )
        z = self.low + (self.high - self.low) * eps
        log_p = T.full((num_samples,), self.log_prob_val, device=z.device)
        return z, log_p

    def log_prob(self, z, context=None) -> T.Tensor:  # noqa: ARG002
        log_p = T.full((z.shape[0],), self.log_prob_val, device=z.device)
        out_range = (z < self.low) | (self.high < z)
        ind_inf = T.any(out_range.view(z.shape[0], -1), dim=-1)
        log_p[ind_inf] = -T.inf
        return log_p


def zero_flat(z: T.Tensor) -> T.Tensor:
    return T.zeros(z.shape[0], dtype=z.dtype, device=z.device)


class PermuteEvenOdd(nf.flows.Flow):
    """Permutation features along the channel dimension swapping even and odd values."""

    def __init__(self) -> None:
        super().__init__()

    def _permute(self, z: T.Tensor) -> T.Tensor:
        z1 = z[:, 0::2]
        z2 = z[:, 1::2]
        return T.stack((z2, z1), dim=2).view(z.shape[0], -1)

    def forward(self, z, context=None) -> tuple:  # noqa: ARG002
        return self._permute(z), zero_flat(z)

    def inverse(self, z, context=None) -> tuple:  # noqa: ARG002
        return self._permute(z), zero_flat(z)


class LULinear(nf.flows.Flow):
    """Invertible linear layer using LU decomposition.

    Needed because normflows doesn't offer this layer by itself.
    Also needed to change the caching mechanism for ONNX export.
    """

    def __init__(self, num_channels: int, identity_init: bool = True):
        super().__init__()
        self.linear = nf.flows.mixing._LULinear(  # noqa: SLF001
            num_channels, identity_init=identity_init
        )

    def use_cache(self, use_cache: bool = True) -> None:
        self.linear.use_cache(use_cache)

    def forward(self, z, context=None) -> tuple[T.Tensor, T.Tensor]:
        z, log_det = self.linear.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None) -> tuple:
        z, log_det = self.linear(z, context=context)
        return z, log_det.view(-1)


class Tanh(nf.flows.Flow):
    """Invertible tanh transformation.

    No liklihood contribution.
    FORWARD IN NORMFLOWS IS Z -> X!!!
    """

    def __init__(self, prescale: float = 1.0):
        super().__init__()
        self.prescale = prescale

    def forward(self, z, context=None) -> tuple[T.Tensor, T.Tensor]:  # noqa: ARG002
        z = T.atanh(z) / self.prescale
        return z, zero_flat(z)

    def inverse(self, z, context=None) -> tuple:  # noqa: ARG002
        z = T.tanh(z * self.prescale)
        return z, zero_flat(z)


class CoupledRationalQuadraticSpline(nf.flows.Flow):
    """Overloaded class from normflows which allow init_identity.

    This is a single coupling layer using rational quadratic splines.
    """

    def __init__(
        self,
        num_input_channels: int,
        num_blocks: int,
        num_hidden_channels: int,
        num_context_channels: int | None = None,
        num_bins: int = 8,
        tails: str = "linear",
        tail_bound: float = 3.0,
        activation: str = "ReLU",
        dropout_probability: float = 0.0,
        reverse_mask: bool = False,
        init_identity: bool = True,
    ) -> None:
        super().__init__()

        # Need to define the network construction function
        def transform_net_create_fn(in_features, out_features):
            # I find that my MLPs use context information better!
            net = MLP(
                inpt_dim=in_features,
                outp_dim=out_features,
                ctxt_dim=num_context_channels or 0,
                hddn_dim=num_hidden_channels,
                num_blocks=num_blocks,
                act_h=activation,
                dropout=dropout_probability,
                ctxt_in_inpt=False,
                ctxt_in_hddn=True,
                ctxt_in_outp=True,  # Want to set this to false in the future
            )

            # For the identity inits with a spline they must follow predefined values
            if init_identity:
                nn.init.constant_(net.output_block.layers[0].weight, 0.0)
                nn.init.constant_(
                    net.output_block.layers[0].bias,
                    np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1),
                )
            return net

        # Create the coupling layer itself
        self.prqct = PiecewiseRationalQuadraticCoupling(
            mask=create_alternating_binary_mask(num_input_channels, even=reverse_mask),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            apply_unconditional_transform=True,
            # This allows the non-transformed values to still be modified by a spline
        )

    def forward(self, z, context=None) -> tuple:
        z, log_det = self.prqct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None) -> tuple:
        z, log_det = self.prqct(z, context)
        return z, log_det.view(-1)


def rqs_flow(
    xz_dim: int,
    ctxt_dim: int = 0,
    num_stacks: int = 3,
    mlp_width: int = 32,
    mlp_depth: int = 2,
    mlp_act: Callable = nn.LeakyReLU,
    tail_bound: float = 4.0,
    dropout: float = 0.0,
    num_bins: int = 8,
    do_lu: bool = True,
    init_identity: bool = True,
    tanh_prescale: float | None = None,
    do_norm: bool = False,
    base_dist: Literal["gaussian", "uniform"] = "gaussian",
    flow_type: Literal["autoregressive", "coupling"] = "coupling",
) -> nf.NormalizingFlow | nf.ConditionalNormalizingFlow:
    """Construct a rational quadratic spline normalising flow.

    Parameters
    ----------
    xz_dim : int
        The dimensionality of the input and output of the flow.
    ctxt_dim : int, optional
        The dimensionality of the context input to the flow. By default 0.
    num_stacks : int, optional
        The number of coupling layers to stack. By default 3.
    mlp_width : int, optional
        The width of the hidden layers in the coupling network. By default 32.
    mlp_depth : int, optional
        The depth of the hidden layers in the coupling network. By default 2.
    mlp_act : Callable, optional
        The activation function to use in the coupling network. By default nn.LeakyReLU.
    tail_bound : float, optional
        The bound on the tails of the spline. By default 4.0.
    dropout : float, optional
        The dropout probability in the coupling network. By default 0.0.
    num_bins : int, optional
        The number of bins in the spline. By default 8.
    do_lu : bool, optional
        Whether to use LU decomposition in the coupling layers. By default True.
        WARNING: This is not supported in ONNX export.
    init_identity : bool, optional
        Whether to initialise the coupling layers as the identity. By default True.
        Strongly recommended for stability.
    tanh_prescale : float, optional
        Whether to prescale the input with a tanh function. By default None.
    do_norm : bool, optional
        Whether to use activation normalisation in the flow. By default False.
    base_dist : str, optional
        The base distribution to use. By default "gaussian".
    flow_type : str
        The type of flow to use. By default "coupling".
    """
    assert flow_type in {"autoregressive", "coupling"}

    # Set the kwargs for the flow as expected by normflows
    kwargs = {
        "num_input_channels": xz_dim,
        "num_blocks": mlp_depth,
        "num_hidden_channels": mlp_width,
        "num_context_channels": ctxt_dim or None,
        "num_bins": num_bins,
        "tail_bound": tail_bound,
        "activation": mlp_act,
        "dropout_probability": dropout,
        "init_identity": init_identity,
    }

    # For MADE we need to use the predefined autoregressive flow which means that
    # We need permutation layers between each
    if flow_type == "autoregressive":
        fn = nf.flows.AutoregressiveRationalQuadraticSpline
        perm = nf.flows.LULinearPermute if do_lu else nf.flows.Permute
        kwargs["activation"] = getattr(nn, mlp_act)  # Their net needs a class

    # For coupling layers we use our overloaded class and instead of permutation
    # The mask is alternated when building each layer
    elif flow_type == "coupling":
        fn = CoupledRationalQuadraticSpline
        perm = LULinear if do_lu else None

    # Build the flow
    flows = []

    for i in range(num_stacks):
        # For coupling layers we need to alternate the mask instead of permuting
        if flow_type == "coupling":
            kwargs["reverse_mask"] = i % 2 == 1

        # Add the flow, as well as the permutation layer and normalisation if needed
        flows += [fn(**kwargs)]
        if perm is not None:
            flows += [perm(xz_dim)]
        if do_norm:
            flows += [nf.flows.ActNorm(xz_dim)]

    # Initial prescaling layers
    if tanh_prescale is not None:
        flows += [Tanh(prescale=tanh_prescale)]

    # Set base distribuiton
    if base_dist == "gaussian":
        q0 = DiagGaussian(xz_dim, trainable=False)
    elif base_dist == "uniform":
        q0 = Uniform(xz_dim)

    # Return the full flow
    if ctxt_dim:
        return nf.ConditionalNormalizingFlow(q0=q0, flows=flows)
    return nf.NormalizingFlow(q0=q0, flows=flows)


def prepare_for_onnx(
    flowwrapper: nn.Module,
    dummy_input: Any,
    method: str = "sample",
) -> None:
    """Prepare a flow for export to ONNX primarily by filling the LU cache."""
    flowwrapper.eval()

    # Switch to cache mode
    n_changed = 0
    for module in base_modules(flowwrapper):
        try:
            module.use_cache(True)
            n_changed += 1
        except AttributeError:
            pass
    print(f"Switched {n_changed} modules to cache mode")

    # Call the method to fill the cache
    if isinstance(dummy_input, tuple):
        getattr(flowwrapper, method)(*dummy_input)
    else:
        getattr(flowwrapper, method)(dummy_input)

    # Remove gradients from the LU caches layers
    n_changed = 0
    for module in base_modules(flowwrapper):
        try:
            module.cache.inverse = module.cache.inverse.data
            module.cache.logabsdet = module.cache.logabsdet.data
            n_changed += 1
        except AttributeError:
            pass
    print(f"Removed cache gradients from {n_changed} modules")
