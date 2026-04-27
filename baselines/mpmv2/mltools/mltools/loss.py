"""Custom loss functions and methods to calculate them."""

import torch as T
import torch.nn.functional as F
from torch import nn


@T.compile
def sigmoid_focal_loss(
    inputs: T.Tensor,
    targets: T.Tensor,
    gamma: float = 2,
    pos_weight: float = 1,
    reduction: str = "mean",
) -> T.Tensor:
    """Focal loss for imbalanced binary classification.

    Parameters
    ----------
    inputs : T.Tensor
        The input tensor from the model
    targets : T.Tensor
        The target tensor for the model
    gamma : float, optional
        The gamma value for the focal loss, by default 2
    pos_weight : float, optional
        The positive class weight, by default 1
    reduction : str, optional
        The reduction method for the loss, by default "mean"
    """
    inputs = inputs.float()
    targets = targets.float()

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p = T.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if pos_weight != 1:
        weight = 1 + (pos_weight - 1) * targets
        loss = weight * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def bce_with_label_smoothing(
    output: T.Tensor, target: T.Tensor, smoothing: float = 0.05, **kwargs
) -> T.Tensor:
    """Calculate binary cross entropy with label smoothing."""
    if smoothing > 0:
        target = target - (target - 0.5).sign() * T.rand_like(target) * smoothing
    return F.binary_cross_entropy_with_logits(output, target.view_as(output), **kwargs)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification."""

    def __init__(
        self, gamma: float = 2, pos_weight: float = 1, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: T.Tensor, targets: T.Tensor) -> T.Tensor:
        return sigmoid_focal_loss(
            inputs, targets, self.gamma, self.pos_weight, self.reduction
        )


def contrastive_loss(x1: T.Tensor, x2: T.Tensor, temperature: float = 0.2) -> T.Tensor:
    """Calculate standard contrastive loss between two sets of embeddings.

    Remember there exist negative samples within x1. Not just between x1 and x2.
    """
    # Concatenate and get the similarity matrix
    batch_size = x1.shape[0]
    z = T.cat([x1, x2])
    sim = F.cosine_similarity(z.unsqueeze(0), z.unsqueeze(1), dim=2) / temperature

    # Fill in the matrix with negative infinities along the diagonal
    mask = T.eye(batch_size * 2, device=x1.device, dtype=bool)
    sim[mask] = -T.inf

    # Calculate the loss using the reduced form of nce
    return -T.diag(sim, batch_size).mean() + T.logsumexp(sim, dim=-1).mean()


@T.autocast("cuda", enabled=False)
def koleo_loss(x: T.Tensor, eps: float = 1e-8, normed: bool = False) -> T.Tensor:
    """Kozachenko-Leonenko entropic loss regularizer.

    From Sablayrolles et al. - 2018 - Spreading vectors for similarity search

    Parameters
    ----------
    x : T.Tensor
        The input tensor to calculate the Kozachenko-Leonenko entropy
        Must be of shape (batch, features)
    eps : float, optional
        The epsilon value to avoid numerical instability, by default 1e-8
    normed : bool, optional
        If the input tensor is already normalized, by default False
    """
    # Normalize the input if not already
    if not normed:
        x = F.normalize(x, eps=eps, dim=-1)

    # Calculate the closest pair idxes via the max inner product
    with T.no_grad():
        dots = T.mm(x, x.t())
        dots.fill_diagonal_(-1)
        min_idx = T.argmax(dots, dim=1)

    # Get the distance between closest pairs
    distances = F.pairwise_distance(x, x[min_idx])

    # Return the kozachenko-leonenko entropy
    return -T.log(distances + eps).mean()


def pressure_loss(x: T.Tensor, normed: bool = False) -> T.Tensor:
    """Positive pressure loss regularizer."""
    if not normed:
        x = F.normalize(x, dim=-1)
    return -F.pdist(x).mean()


def champfer_loss(
    x: T.Tensor, x_mask: T.BoolTensor, y: T.Tensor, y_mask: T.BoolTensor
) -> T.Tensor:
    """Return the champfer loss between two masked point clouds."""
    # Only works if the last dimension is the same
    assert x.shape[-1] == y.shape[-1]

    # Calculate the distance matrix (squared) between the outputs and targets
    dist_matrix = T.cdist(x, y)

    # Ensure the distances between fake nodes take some padding value
    matrix_mask = x_mask.unsqueeze(-1) & y_mask.unsqueeze(-2)
    dist_matrix = dist_matrix.masked_fill(~matrix_mask, 1e8)

    # Get the sum of the minimum along each axis
    min_x = T.min(dist_matrix, dim=-1)[0] * x_mask  # Zeros out the padded
    min_y = T.min(dist_matrix, dim=-2)[0] * y_mask

    # Add the two metrics together (no batch reduction)
    return 0.5 * (T.sum(min_x, dim=-1) + T.sum(min_y, dim=-1)).mean()


def kld_to_norm_loss(means: T.Tensor, log_stds: T.Tensor) -> T.Tensor:
    """Calculate the KL-divergence to a unit normal distribution."""
    return (0.5 * (means * means + (2 * log_stds).exp() - 2 * log_stds - 1)).mean()


class MyBCEWithLogit(nn.Module):
    """A wrapper for the calculating BCE using logits.

    Makes the syntax consistant with pytorch's CrossEntropy loss.
    - BCE wants identical shapes (batch x output)
    - CE wants targets just as indices (batch)

    Applies the following transformations:
    - Automatically squeezes out the batch dimension to ensure same shape
    - Automatically changes targets to floats
    - Automatically puts anything nonzero into class 1

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(*args, **kwargs)

    def forward(self, outputs: T.Tensor, targets: T.Tensor) -> T.Tensor:
        return self.loss_fn(outputs.squeeze(dim=-1), (targets != 0).float())
