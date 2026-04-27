import torch
import torch.nn.functional as F


def contrastive_loss(x_i, x_j, temperature):
    """NT-Xent contrastive loss (InfoNCE).

    Replaces the original O(2B * 2B * D) cosine_similarity broadcast with a
    simple (2B, 2B) matmul on already-normalized vectors, then delegates to
    F.cross_entropy for numerically-stable log-softmax.
    """
    B = x_i.shape[0]
    z_i = F.normalize(x_i, dim=1)
    z_j = F.normalize(x_j, dim=1)
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

    # (2B, 2B) similarity matrix — matmul is O(2B * 2B * D) flops but
    # only keeps a (2B, 2B) result, not a (2B, 2B, D) intermediate.
    sim = torch.mm(z, z.T) / temperature  # (2B, 2B)

    # Mask out self-similarity (diagonal) so it never acts as a negative.
    sim.fill_diagonal_(float("-inf"))

    # Positive pair labels:
    #   row i   (0 ≤ i < B)  → positive at column i + B
    #   row i+B (0 ≤ i < B)  → positive at column i
    labels = torch.cat(
        [
            torch.arange(B, 2 * B, device=x_i.device),
            torch.arange(B, device=x_i.device),
        ]
    )

    return F.cross_entropy(sim, labels)


def align_loss(x, y, alpha=2):
    x_n = F.normalize(x, dim=1)
    y_n = F.normalize(y, dim=1)
    return (x_n - y_n).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    x_n = F.normalize(x, dim=1)
    return torch.pdist(x_n, p=2).pow(2).mul(-t).exp().mean().log()
