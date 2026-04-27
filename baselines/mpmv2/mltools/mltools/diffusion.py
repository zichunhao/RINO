"""A collection of functions for the karras EDM diffusion method."""

import math

import torch as T
from torch import nn
from torchdiffeq import odeint
from tqdm import trange

from .torch_utils import GradsOff, append_dims


@T.no_grad()
def cfm_values(
    x0: T.Tensor,
    do_sigmoid: bool = True,
    time_embedding: nn.Module | None = None,
) -> tuple:
    """Calculate the values needed for continuous flow matching."""
    if do_sigmoid:  # Time values: Batch x 1
        t = T.sigmoid(T.randn(x0.shape[0], 1, device=x0.device))
    else:
        t = T.rand(x0.shape[0], 1, device=x0.device)
    ctxt_t = time_embedding(t) if time_embedding is not None else t
    t = append_dims(t, x0.ndim)
    x1 = T.randn_like(x0)
    xt = (1 - t) * x0 + t * x1
    v = x1 - x0
    return xt, v, t, x1, ctxt_t


def c_values(sigmas: T.Tensor) -> tuple:
    """Calculate the Karras C values.

    Needed to scale the inputs, outputs, and skip connection.
    Assumes your data sigma is 1.
    """
    c_in = 1 / (1 + sigmas**2).sqrt()
    c_out = sigmas / (1 + sigmas**2).sqrt()
    c_skip = 1 / (1 + sigmas**2)
    return c_in, c_out, c_skip


@T.no_grad()
def multistep_consistency_sampling(
    model: nn.Module,
    sigmas: T.Tensor,
    min_sigma: float,
    x: T.Tensor,
    extra_args: dict | None = None,
    same_noise: bool = False,
) -> T.Tensor:
    """Perform multistep consistency sampling from a consistency model.

    Parameters
    ----------
    model : nn.Module
        The model to generate samples from.
    sigmas : torch.Tensor
        The sequence of noise levels to generate samples.
    min_sigma : float
        The minimum noise level.
    x : torch.Tensor
        The initial noise for generation.
    extra_args : dict or None, optional
        Extra arguments to pass to the model. Default is None.
    same_noise : bool, optional
        Whether to use the same noise for each step or not. Default is False.

    Returns
    -------
    x : torch.Tensor
        The final sample.
    """
    extra_args = extra_args or {}
    sigma_shape = x.new_ones([x.shape[0], 1])

    noise = T.randn_like(x)
    x = model(x, sigmas[0] * sigma_shape, **extra_args)
    for sigma in sigmas:
        if not same_noise:
            noise = T.randn_like(x)
        x_t = x + (sigma**2 - min_sigma**2).sqrt() * noise
        x = model(x_t, sigma * sigma_shape, **extra_args)
    return x


@T.no_grad()
def log_likelihood(
    model: nn.Module,
    x: T.Tensor,
    sigmas: T.Tensor,
    extra_args: dict | None = None,
    atol: float = 1e-3,
    rtol: float = 5e-2,
    solver: str = "dopri5",
    mask: T.Tensor | float = 1.0,
) -> tuple:
    """Calculate the liklihood of a batch of data given a diffusion model."""
    # Default dict arguments
    extra_args = extra_args or {}

    # Some starting variables
    fevals = 0
    sigma_shape = x.new_ones([x.shape[0]])

    # Define the funciton for calculating the gradient and trace at each step
    def ode_fn(sigma: float, x: tuple) -> tuple:
        nonlocal fevals
        with T.enable_grad():
            # The solver input is actually a tuple of tensors, the first is the data
            x = x[0].detach().requires_grad_()

            # Like when solving get the denoised output and use to define gradient
            with GradsOff(model):
                denoised = model(x, sigma * sigma_shape, **extra_args)
            d = to_d(x, sigma, denoised) * mask

            # Increment the number of function evaluations used in the solver
            fevals += 1

            # Get the trace estimate using the Skilling-Hutchinson method
            eps = T.randint_like(x, 2) * 2 - 1
            grad = T.autograd.grad(d, x, eps)[0] * eps
            d_ll = grad.flatten(1).sum(1)

        # Return both the gradient for the ode and the trace for the log-liklihood
        return d.detach(), d_ll

    # The input to ode_fn is the data and the starting liklihood (0)
    x_min = x, x.new_zeros([x.shape[0]])

    # Run the solver using the above equation
    sol = odeint(
        ode_fn,
        x_min,
        sigmas,
        atol=atol,
        rtol=rtol,
        method=solver,
    )

    # Pull out the final estimate and the total log lik from the solution
    latent, delta_ll = sol[0][-1], sol[1][-1]
    ll_prior = T.distributions.Normal(0, sigmas[-1]).log_prob(latent) * mask
    ll_prior = ll_prior.flatten(1).sum(1)

    return ll_prior + delta_ll, {"fevals": fevals}


def logsumexp(x: T.Tensor, do_max: bool = True) -> T.Tensor:
    """Apply the log sum exp trick for numerical precision.

    https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    c = x.max() if do_max else x.min()
    return c + T.log(T.sum(T.exp(x - c)))


@T.no_grad()
def shift_gaussian(x: T.Tensor, mu: T.Tensor, var: T.Tensor) -> T.Tensor:
    """Gaussian function shifted such that the max is always 1."""
    diff = (x - mu) ** 2
    diff = diff.sum(dim=tuple(range(2, diff.dim())), keepdims=True)
    diff /= -2 * var
    diff_max = T.max(diff, dim=1, keepdim=True)
    return T.exp(diff - diff_max)


@T.no_grad()
def ideal_denoise(noisy_data, data, sigma):
    """Get the ideal denoised target using a kde of your dataset."""
    gaus_term = shift_gaussian(
        noisy_data.unsqueeze(1),
        data.unsqueeze(0),
        append_dims(sigma, noisy_data.dim() + 1) ** 2,
    )
    numerator = (gaus_term * data.unsqueeze(0)).sum(1)
    return numerator / gaus_term.sum(1)


@T.no_grad()
def one_step_ideal(x, data, sigma_start, sigma_end):
    """Apply just one step of the ideal solver using the euler method."""
    denoised = ideal_denoise(x, data, sigma_start)
    d = (x - denoised) / append_dims(sigma_start, x.dim())
    dt = append_dims((sigma_end - sigma_start), x.dim())
    return x + d * dt


def to_d(x, sigma, denoised):
    """Convert a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@T.no_grad()
def one_step_dpm_2(model, x, sigma_start, sigma_end, extra_args):
    """Apply just one step of the DPM2 to get two adjacent points on the PF- ODE."""
    # Initial setup
    extra_args = extra_args or {}

    # Denoise the sample, and calculate derivative
    denoised = model(x, sigma_start, **extra_args)
    d = to_d(x, sigma_start, denoised)

    # Get the midpoint sigma between the start and end
    sigma_mid = sigma_start.log().lerp(sigma_end.log(), 0.5).exp()
    dt_1 = append_dims(sigma_mid - sigma_start, x.dim())
    dt_2 = append_dims(sigma_end - sigma_start, x.dim())

    # DPM2 2nd order method
    x_2 = x + d * dt_1
    denoised_2 = model(x_2, sigma_mid, **extra_args)
    d_2 = to_d(x_2, sigma_mid, denoised_2)
    return x + d_2 * dt_2


@T.no_grad()
def one_step_heun(model, x, sigma_start, sigma_end, extra_args):
    """Apply just one step of the heun-solver."""
    # Initial setup
    extra_args = extra_args or {}

    # Denoise the sample, and calculate derivative and the time step
    denoised = model(x, sigma_start, **extra_args)
    d = (x - denoised) / append_dims(sigma_start, x.dim())
    dt = append_dims((sigma_end - sigma_start), x.dim())
    x_2 = x + d * dt

    # Heun's 2nd order method
    denoised_2 = model(x_2, sigma_end, **extra_args)
    d_2 = (x_2 - denoised_2) / append_dims(sigma_end, x.dim())
    d_prime = (d + d_2) / 2
    return x + d_prime * dt


def get_sigmas_karras(
    sigma_min: float,
    sigma_max: float,
    n_steps: int = 100,
    rho: float = 7,
) -> T.Tensor:
    """Construct sigmas for the Karras et al schedule.

    Parameters
    ----------
    sigma_min:
        The minimum/final time
    sigma_max:
        The maximum/starting time
    n_steps:
        The number of time steps
    rho:
        The degree of curvature, rho=1 equal step size, recommened 7 for diffusion
    """
    ramp = T.linspace(0, 1, n_steps)
    max_inv_rho = sigma_max ** (1 / rho)
    min_inv_rho = sigma_min ** (1 / rho)
    return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho


@T.no_grad()
def sample_heun(
    model,
    x: T.Tensor,
    sigmas: T.Tensor,
    do_heun_step: bool = True,
    keep_all: bool = False,
    extra_args: dict | None = None,
    disable: bool | None = True,
) -> None:
    """Deterministic sampler using Heun's second order method.

    Parameters
    ----------
    model : nn.Module
        The model to generate samples from.
    x : Tensor
        The initial noise for generation.
    sigmas : Tensor
        The sequence of noise levels to generate samples.
    do_heun_step : bool, optional
        Whether to use Heun's 2nd order method or not. Default is True.
    keep_all : bool, optional
        Whether to store the samples at each step or not. Default is False.
    extra_args : dict or None, optional
        Extra arguments to pass to the model. Default is None.
    disable : bool or None, optional
        Whether to disable the progress bar or not. Default is True.


    Returns
    -------
    tuple
        A tuple containing two elements.
        - The generated samples.
        - All the intermediate samples (if keep_all is True), otherwise None.

    Notes
    -----
    Hard coded such that t = sigma and s(t) = 1.
    Alg. 1 from the https://arxiv.org/pdf/2206.00364.pdf.
    """
    # Initial setup
    num_steps = len(sigmas) - 1
    all_stages = [x] if keep_all else None
    sigma_shape = x.new_ones([x.shape[0], 1])
    extra_args = extra_args or {}

    # Start iterating through each timestep
    for i in trange(num_steps, disable=disable):
        # Denoise the sample, and calculate derivative and the time step
        denoised = model(x, sigmas[i] * sigma_shape, **extra_args)
        d = (x - denoised) / sigmas[i]
        dt = sigmas[i + 1] - sigmas[i]

        # Apply the integration step
        if not do_heun_step or sigmas[i + 1] == 0:
            # Euler step (=DDIM with this noise schedule)
            x = x + d * dt
        else:
            # Heun's 2nd order method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * sigma_shape, **extra_args)
            d_2 = (x_2 - denoised_2) / sigmas[i + 1]
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt

        # Update the track
        if keep_all:
            all_stages.append(x)

    if keep_all:
        return x, all_stages
    return x


@T.no_grad()
def sample_stochastic_heun(
    model,
    x: T.Tensor,
    sigmas: T.Tensor,
    do_heun_step: bool = True,
    keep_all: bool = False,
    s_churn: float = 40.0,
    s_tmin: float = 0.05,
    s_tmax: float = 50.0,
    s_noise: float = 1.003,
    extra_args: dict | None = None,
    disable: bool | None = True,
) -> None:
    """Stochastic sampler using Heun's second order method.

    Parameters
    ----------
    model : nn.Module
        The model to generate samples from.
    x : Tensor
        The initial noise for generation.
    sigmas : Tensor
        The sequence of noise levels to generate samples.
    do_heun_step : bool, optional
        Whether to use Heun's 2nd order method or not. Default is True.
    keep_all : bool, optional
        Whether to store the samples at each step or not. Default is False.
    s_churn : float, optional (default=40.0)
        Changes the time for the iteration by a small amount
    s_tmin : float, optional (default=0.05)
        The lower bound of sigma where the stochasticity is allowed
    s_tmax : float, optional (default=50.0)
        The upper bound of sigma where the stochasticity is allowed
    s_noise : float, optional (default=1.003)
        The std of the noise which is added to the sample
    extra_args : dict or None, optional
        Extra arguments to pass to the model. Default is None.
    disable : bool or None, optional
        Whether to disable the progress bar or not. Default is True.

    Returns
    -------
    tuple
        A tuple containing two elements.
        - The generated samples.
        - All the intermediate samples (if keep_all is True), otherwise None.

    Notes
    -----
    - Equivalent to the deterministic case if s_churn = 0
    - Alg. 2 from the https://arxiv.org/pdf/2206.00364.pdf
    - Hard coded such that t = sigma and s(t) = 1
    - Default s values are taken from empirical results in the paper
    """
    # Initial setup
    num_steps = len(sigmas) - 1
    all_stages = [x] if keep_all else None
    sigma_shape = x.new_ones([x.shape[0], 1])
    extra_args = extra_args or {}

    # Start iterating through each timestep
    for i in trange(num_steps, disable=disable):
        # Get gamma factor (time perturbation)
        gamma = (
            min(s_churn / num_steps, math.sqrt(2.0) - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )

        # Shift the sigma value and the sample using the noise based gamma
        sigma_hat = sigmas[i] * (1 + gamma)

        # Inject noise into x if the gamma value is above zero
        if gamma > 0:
            eps = T.randn_like(x) * s_noise
            x = x + eps * math.sqrt(sigma_hat**2 - sigmas[i] ** 2)

        # Denoise the sample, and calculate derivative and the time step
        denoised = model(x, sigma_hat * sigma_shape, **extra_args)
        d = (x - denoised) / sigma_hat
        dt = sigmas[i + 1] - sigma_hat

        # Apply the integration step
        if not do_heun_step or sigmas[i + 1] == 0:
            # Euler step (=DDIM with this noise schedule)
            x = x + d * dt
        else:
            # Heun's 2nd order method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * sigma_shape, **extra_args)
            d_2 = (x_2 - denoised_2) / sigmas[i + 1]
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt

        # Update the track
        if keep_all:
            all_stages.append(x)

    if keep_all:
        return x, all_stages
    return x
