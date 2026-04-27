"""Stuff for bayesian neural networks."""

import math

import torch as T
import torch.nn.functional as F
from torch import nn


def contains_bayesian_layers(model: nn.Module) -> bool:
    """Check if a network has at least one BayesianLinear layer.

    Loops over a network's submodules and looks for BayesianLinear layers.
    """
    if isinstance(model, BayesianLinear):
        return True
    return any(isinstance(m, BayesianLinear) for m in model.modules())


def prior_loss(model: nn.Module) -> T.Tensor | int:
    """Calculate the prior loss of a bayesian neural network."""
    kl_loss = model.prior_kl() if isinstance(model, BayesianLinear) else 0
    for m in model.children():
        kl_loss = kl_loss + prior_loss(m)
    return kl_loss


def change_deterministic(model: nn.Module, setting: bool = True) -> None:
    """Change a bayesian neural network to be deterministic/stochastic."""
    if isinstance(model, BayesianLinear):
        model.deterministic = setting
    for m in model.children():
        change_deterministic(m, setting)


class BayesianLinear(nn.Module):
    """A bayesian linear layer.

    Here every single weight in the matrix is
    modeled as a gaussian and sampled during each forward pass. The biases
    however are NOT noisy and are kept deterministic This layer uses the local
    parameterisation trick explained here during training:
    https://arxiv.org/pdf/1506.02557.pdf.

    - This saves computation time, as we are not sampling noise for every weight
    - It also helps with stability, as the gradient updates observe less variance

    The bias and the (nominal) weight values are labeled the same as nn.Linear
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sig: float = 1.0,
        deterministic: bool = False,
        logsig2_init: float = -9.0,
    ):
        """Parameters
        ----------
        in_features:
            The size of the input tensor
        out_features:
            The size of the output tensor
        prior_sig:
            The width of the prior for the weights
        deterministic:
            If the network should be purely deterministic
        logsig2_init:
            The starting means for logsig2, small to give network chance
        """
        super().__init__()

        # Save the class attributes
        self.n_in = in_features
        self.n_out = out_features
        self.deterministic = deterministic
        self.prior_sig = prior_sig
        self.logsig2_init = logsig2_init

        # The learnable parameters of the model
        self.bias = nn.Parameter(T.empty(self.n_out))
        self.weight = nn.Parameter(T.empty(self.n_out, self.n_in))
        self.w_logsig2 = nn.Parameter(T.empty(self.n_out, self.n_in))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the learnable parameters, biases are initialised as zeros."""
        self.bias.data.data.zero_()
        self.weight.data.normal_(0, 1 / math.sqrt(self.n_in))
        self.w_logsig2.data.normal_(self.logsig2_init, 0.001)

    def prior_kl(self) -> T.Tensor:
        """Calculate the KL-divergence between the current weights and the prior."""
        w_logsig2 = self.w_logsig2.clamp(-11, 11)  # For numerical stability
        return (
            0.5
            * (
                self.prior_sig * (self.weight**2 + w_logsig2.exp())
                - w_logsig2
                - math.log(self.prior_sig)
                - 1
            ).sum()
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass for the layer.

        - Behaves differently depending on tain() or eval() has been called
        """
        # For deterministic mode, simply calculate the nomimal output
        if self.deterministic:
            return F.linear(x, self.weight, self.bias)  # Nominal

        # For numerical stability going forward
        w_logsig2 = self.w_logsig2.clamp(-11, 11)

        # In training mode, we perform the Local Reparam Trick
        if self.training:
            nom_out = F.linear(x, self.weight, self.bias)  # Nominal
            var_out = F.linear(x**2, w_logsig2.exp())
            return nom_out + var_out.sqrt() * T.randn_like(nom_out)

        # In evaluation mode we do the full noise generation
        noise = T.exp(self.w_logsig2 / 2) * T.randn_like(self.weight)
        return F.linear(x, self.weight + noise, self.bias)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.n_in}, {self.n_out})"
