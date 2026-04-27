"""Highly configurable MLP for all needs."""

import itertools

import torch as T
from torch import nn

from .bayesian import BayesianLinear
from .torch_utils import zero_module


class MLPBlock(nn.Module):
    """A simple MLP block that makes up a dense network.

    Made up of several layers containing:
    - linear map
    - activation function [Optional]
    - layer normalisation [Optional]
    - dropout [Optional]

    Only the input of the block is concatentated with context information.
    For residual blocks, the input is added to the output of the final layer.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        num_layers: int = 1,
        activation: str | None = "lrlu",
        norm: str | None = None,
        dropout: float = 0,
        do_residual: bool = False,
        do_bayesian: bool = False,
        init_zeros: bool = False,
    ) -> None:
        """Init method for MLPBlock.

        Parameters
        ----------
        inpt_dim : int
            The number of features for the input layer
        outp_dim : int
            The number of output features
        ctxt_dim : int, optional
            The number of contextual features to concat to the inputs, by default 0
        num_layers : int, optional
            The number of transform layers in this block, by default 1
        activation : str, optional
            A string indicating the name of the activation function, by default "lrlu"
        norm : str, optional
            A string indicating the name of the normalisation, by default None
        dropout : float, optional
            The dropout probability, 0 implies no dropout, by default 0
        do_residual : bool, optional
            Add to previous output, only if dim does not change, by default 0
        do_bayesian : bool, optional
            If to fill the block with bayesian linear layers, by default False
        init_zeros : bool, optional
            If the final layer weights and bias values are set to zero
            Does not apply to bayesian layers
        """
        super().__init__()

        # Save the input and output dimensions of the module
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.init_zeros = init_zeros
        self.do_res = do_residual and (inpt_dim == outp_dim)

        # Change certain defaults if using bayesian layers
        linear = BayesianLinear if do_bayesian else nn.Linear
        init_zeros = init_zeros and not do_bayesian

        # Initialise the block layers as a module list
        self.layers = nn.ModuleList()
        for n in range(num_layers):
            # Increase the input dimension of the first layer to include context
            lyr_in = inpt_dim + ctxt_dim if n == 0 else outp_dim

            # Linear transform, activation, normalisation, dropout
            self.layers.append(linear(lyr_in, outp_dim))

            # Check if the linear layer should be initialised with zeros
            with_zeros = init_zeros and n == num_layers - 1
            if with_zeros:
                zero_module(self.layers[-1])

            # Add the activation layer
            if activation is not None:
                self.layers.append(getattr(nn, activation)())

            # Normalisation layer, not right after initialising with zeros
            if norm is not None and not with_zeros:
                self.layers.append(getattr(nn, norm)(outp_dim))

            # Dropout layer
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        """Pass the input through the block and return the output."""
        if self.do_res:  # Double checked that this does copy
            orig = x
        if self.ctxt_dim:  # Concatenate context to input
            x = T.cat([x, ctxt], dim=-1)
        for layer in self.layers:  # Pass through each layer
            x = layer(x)
        if self.do_res:  # Add the original input to the output
            x = x + orig
        return x

    def __repr__(self) -> str:
        """Generate a one line string summing up the components of the block."""
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        for b in self.layers:
            string += "->" + str(b).split("(", 1)[0]
            if isinstance(b, nn.Linear) and self.init_zeros and b in self.layers[-4:]:
                string += "(zero)"
        string += "->" + str(self.outp_dim)
        if self.do_res:
            string += "(add)"
        return string


class MLP(nn.Module):
    """A dense neural network made from a series of consecutive MLP blocks.

    Supports context injection layers.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        hddn_dim: int | list = 32,
        num_blocks: int = 1,
        num_layers_per_block: int = 1,
        act_h: str = "SiLU",
        act_o: str | None = None,
        norm: str | None = None,
        dropout: float = 0,
        drop_on_output: bool = False,
        norm_on_output: bool = False,
        do_res: bool = False,
        ctxt_in_inpt: bool = True,
        ctxt_in_hddn: bool = False,
        ctxt_in_outp: bool = False,
        do_bayesian: bool = False,
        init_zeros: bool = False,
    ) -> None:
        """Initialise the MLP.

        Parameters
        ----------
        inpt_dim : int
            The number of input features
        outp_dim : int
            The number of output features
        ctxt_dim : int, optional
            The number of context features to inject, by default 0
        hddn_dim : int | list, optional
            The number of hidden features in each block, by default 32
            If a list it will override the num_blocks parameter
        num_blocks : int, optional
            The number of hidden blocks, by default 1
        num_layers_per_block : int, optional
            The number of liner layers in each hidden block, by default 1
        act_h : str, optional
            The activation function for the hidden layers, by default "SiLU"
        act_o : str, optional
            The activation function for the output layer, by default None
        norm : str, optional
            The normalisation layer to use, by default None
        dropout : float, optional
            The dropout probability, by default 0
        drop_on_output : bool, optional
            If to apply dropout to the output layer, by default False
        norm_on_output : bool, optional
            If to apply normalisation to the output layer, by default False
        do_res : bool, optional
            If to add the input to the output of each hidden block, by default False
        ctxt_in_inpt : bool, optional
            If to inject context into the input layer, by default True
        ctxt_in_hddn : bool, optional
            If to inject context into each hidden layer, by default False
        ctxt_in_outp : bool, optional
            If to inject context into the output layer, by default False
        do_bayesian : bool, optional
            If to use bayesian linear layers, by default False
        init_zeros : bool, optional
            If to initialise final layer weights and bias to zero, by default False
        """
        super().__init__()

        # Check that the context is used somewhere
        if ctxt_dim and not ctxt_in_inpt and not ctxt_in_hddn and not ctxt_in_outp:
            raise ValueError("Network has context inputs but nowhere to use them!")

        # Attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.hddn_dim = (
            hddn_dim if isinstance(hddn_dim, list) else num_blocks * [hddn_dim]
        )
        self.num_blocks = len(self.hddn_dim)

        # For compatibility with the normflows package we need this attribute
        self.hidden_features = self.hddn_dim[-1]

        # Input MLP block
        self.input_block = MLPBlock(
            inpt_dim=self.inpt_dim,
            outp_dim=self.hddn_dim[0],
            ctxt_dim=self.ctxt_dim * ctxt_in_inpt,
            activation=act_h,
            norm=norm,
            dropout=dropout,
            do_bayesian=do_bayesian,
        )

        # All hidden blocks as a single module list
        hidden_blocks = [
            MLPBlock(
                inpt_dim=h_1,
                outp_dim=h_2,
                ctxt_dim=self.ctxt_dim * ctxt_in_hddn,
                num_layers=num_layers_per_block,
                activation=act_h,
                norm=norm,
                dropout=dropout,
                do_residual=do_res,
                init_zeros=init_zeros,
                do_bayesian=do_bayesian,
            )
            for h_1, h_2 in itertools.pairwise(self.hddn_dim)
        ]

        # Only wrap with module list if there are blocks
        self.hidden_blocks = nn.ModuleList(hidden_blocks) if hidden_blocks else []

        # Output block
        self.output_block = MLPBlock(
            inpt_dim=self.hddn_dim[-1],
            outp_dim=self.outp_dim,
            ctxt_dim=self.ctxt_dim * ctxt_in_outp,
            activation=act_o,
            do_bayesian=do_bayesian,
            norm=norm if norm_on_output else None,
            dropout=dropout * drop_on_output,
            init_zeros=init_zeros,
        )

    def forward(
        self,
        inputs: T.Tensor,
        ctxt: T.Tensor | None = None,
        context: T.Tensor | None = None,
    ) -> T.Tensor:
        """Pass through the mlp."""
        # Use context as a synonym for ctxt (normflow compatibility)
        ctxt = ctxt if ctxt is not None else context

        # Reshape the context if it is available. Equivalent to performing
        # multiple ctxt.unsqueeze(1) until the dim matches the input.
        if ctxt is not None and (dim_diff := inputs.dim() - ctxt.dim()) > 0:
            ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
            ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        # Pass through all layers
        inputs = self.input_block(inputs, ctxt)
        for h_block in self.hidden_blocks:
            inputs = h_block(inputs, ctxt)
        return self.output_block(inputs, ctxt)

    def __repr__(self):
        string = ""
        string += "\n  (inp): " + repr(self.input_block) + "\n"
        for i, h_block in enumerate(self.hidden_blocks):
            string += f"  (h-{i + 1}): " + repr(h_block) + "\n"
        string += "  (out): " + repr(self.output_block)
        return string
