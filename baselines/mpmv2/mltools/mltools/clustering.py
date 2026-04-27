"""Clustering algorithms for pytorch tensors."""

import torch as T
from tqdm import tqdm, trange


@T.no_grad()
def kmeans(
    x: T.Tensor,
    num_clusters: int,
    tol_per_dimension=6e-5,
    max_iter=1000,
    use_plusplus=False,
) -> T.Tensor:
    """Perform kmeans and return the cluster centers."""
    # Make sure that the input is a float
    x = x.float()
    dataset_size = x.shape[0]

    # Check if we can use the plusplus algorithm
    cant_use_multi = False
    if dataset_size > 2**24:
        cant_use_multi = True
        use_plusplus = False
        print("Warning: Disabling kmeans++ for large datasets.")

    # Check if the number of clusters is larger than the dataset
    if num_clusters > dataset_size:
        raise ValueError("Number of clusters is larger than the dataset size.")

    # Use the kmeans++ algorithm to initialise the cluster centers
    if use_plusplus:
        cluster_centers = T.zeros_like(x[:num_clusters])
        cluster_centers[0] = T.clone(x[0])
        for i in trange(1, num_clusters, desc="Initialising centers"):
            min_dist = T.cdist(x, cluster_centers[:i]).min(dim=-1).values
            probs = min_dist / min_dist.sum()
            cluster_centers[i] = T.clone(x[T.multinomial(probs, 1)])

    # Just step through the dataset and pick some random points
    else:
        indices = T.arange(dataset_size)[:: dataset_size // num_clusters][:num_clusters]
        cluster_centers = T.clone(x[indices])

    # Iterate until convergence
    tqdm_meter = tqdm(desc="Running kmeans")
    for iteration in range(max_iter):
        # Calculate the distance between the inputs and the cluster centers
        dist = T.cdist(x, cluster_centers)

        # For each of the inputs find the closest cluster center
        idxes = T.argmin(dist, dim=-1).unsqueeze(-1)

        # For each cluster, calculate the mean of the inputs assigned to it
        cluster_means = T.zeros_like(cluster_centers)
        cluster_means.scatter_add_(0, idxes.expand_as(x), x)
        cluster_means /= T.bincount(idxes.squeeze(), minlength=num_clusters).unsqueeze(
            -1
        )

        # If there are empty clusters, replace them with a random input
        empty_clusters = T.isnan(cluster_means).any(dim=-1)
        if empty_clusters.any():
            if cant_use_multi:
                new_idx = T.randint(0, dataset_size, (empty_clusters.sum(),))
            else:
                new_idx = T.multinomial(
                    T.ones(dataset_size), empty_clusters.sum(), replacement=False
                )
            cluster_means[empty_clusters] = T.clone(x[new_idx])

        # Calculate the total shift per dimension of all cluster centers
        shift = T.norm(cluster_means - cluster_centers, dim=-1).sum()
        shift /= x.shape[-1]

        # Replace the cluster centers with the new means
        cluster_centers = cluster_means

        # Update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f"{iteration}",
            center_shift=f"{shift:0.6f}",
            tol_per_dimension=f"{tol_per_dimension:0.6f}",
        )
        tqdm_meter.update()
        if shift < tol_per_dimension:
            break

    tqdm_meter.close()

    # Print the cluster occupancy
    occupancy = T.bincount(idxes.squeeze(), minlength=num_clusters)
    print("Finished kmeans clustering with occupancy")
    print(f" - min={occupancy.min()}")
    print(f" - mean={occupancy.float().mean():.0f}")
    print(f" - max={occupancy.max()}")

    # Print a warning if the iteration limit was reached
    if iteration == max_iter - 1:
        print("Warning: kmeans reached maximum iterations without convergence.")

    return cluster_means
