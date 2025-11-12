import torch


def get_cov_loss(z: torch.Tensor, z_mean: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the covariance loss.

    Args:
        z: The input tensor, either shape (batch_size, feature_dim) or
           (num_views, batch_size, feature_dim)
        z_mean: Optional pre-computed mean tensor.

    Returns:
        The covariance loss.
    """
    if z.dim() == 3:
        num_views, N, D = z.shape
        if z_mean is None:
            z_mean = z.mean(dim=1, keepdim=True)  # [num_views, 1, D]

        z_cent = z - z_mean
        # Covariance for each view
        cov = (z_cent.transpose(1, 2) @ z_cent) / (N - 1)  # [num_views, D, D]
        diag = torch.eye(D, dtype=bool, device=z.device)
        cov_off_diag = cov * ~diag

        # Average loss across views
        return torch.pow(cov_off_diag, 2).sum(dim=(1, 2)).mean().div(D)
    else:
        N, D = z.shape

        if z_mean is None:
            z_mean = z.mean(dim=0)
        z_cent = z - z_mean
        cov = (z_cent.T @ z_cent) / (N - 1)
        diag = torch.eye(D, dtype=bool, device=z.device)
        cov_off_diag = cov * ~diag

        return torch.pow(cov_off_diag, 2).sum().div(D)
