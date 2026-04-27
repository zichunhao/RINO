import torch


def apply_matrices(
    x: torch.Tensor,
    matrices: torch.Tensor,
    idx_start: int = 0,
    in_place: bool = False,
) -> torch.Tensor:
    """
    Apply a set of matrix transformations to a set of higher-rank tensors.

    Args:
        x: The tensors to transform, shape (N_x, ..., D_x).
        matrices: The transformation matrices, shape (N_x, D_out, D_in).
        idx_start: The starting index of the vectors to transform. Default is 0.
        in_place: Whether to apply the transformations in place. Default is False.

    Returns:
        The tensor with the corresponding entries transformed by the matrices.
    """
    N_x, D_out, D_in = matrices.shape
    *x_shape, D_x = x.shape

    if x_shape[0] != N_x:
        raise ValueError(
            "Expected the same number of elements in the first dimension of x and matrices. "
            f"Found {x_shape[0]} in x and {N_x} in matrices."
        )
    if idx_start + D_in > D_x:
        raise IndexError(
            f"Index out of bounds. Expected idx_start + D_in <= D_x. Found {idx_start} + {D_in} > {D_x}."
        )

    if not in_place:
        x = x.clone()

    # Reshape x to (N_x, -1, D_x) for batch matrix multiplication
    x_reshaped = x.view(N_x, -1, D_x)

    # Transform the vectors
    vectors = x_reshaped[..., idx_start : idx_start + D_in]
    transformed = torch.bmm(matrices, vectors.transpose(-1, -2)).transpose(-1, -2)

    # Update the vectors
    x_reshaped[..., idx_start : idx_start + D_out] = transformed

    # Reshape x back to its original shape
    x = x_reshaped.view(*x_shape, D_x)

    return x
