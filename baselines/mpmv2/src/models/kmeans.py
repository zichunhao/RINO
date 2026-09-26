"""Pure-PyTorch K-means codebook for the masked K-means task.

``KMeansCodebook`` exposes the interface of the ``torchpq.clustering.KMeans``
objects pickled in ``resources/`` (``n_clusters``, ``centroids`` of shape
``(dim, n_clusters)``, ``predict`` on ``(dim, n_points)`` inputs and a
``weights`` buffer with inverse-frequency class weights), so ``KmeansTask`` and
the other models load either kind of codebook in the same way. It needs neither
``torchpq`` nor ``cupy`` and runs on CPU as well as GPU.
"""

import logging

import torch as T
from torch import nn

log = logging.getLogger(__name__)


class KMeansCodebook(nn.Module):
    """K-means codebook with chunked nearest-centroid assignment.

    Parameters
    ----------
    n_clusters : int
        Number of centroids.
    dim : int
        Dimension of the clustered features.
    chunk_size : int, optional
        Number of points assigned per chunk. Each chunk materialises a
        ``(chunk_size, n_clusters)`` float32 distance matrix, so this bounds the
        peak memory of ``predict`` and ``fit``.
    """

    def __init__(self, n_clusters: int, dim: int, chunk_size: int = 8192) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.chunk_size = chunk_size
        self.register_buffer("centroids", T.zeros(dim, n_clusters))
        self.register_buffer("weights", T.ones(n_clusters))

    @T.no_grad()
    def predict(self, x: T.Tensor) -> T.Tensor:
        """Return the index of the nearest centroid for each column of ``x``.

        Parameters
        ----------
        x : T.Tensor
            Points of shape ``(dim, n_points)``, as for ``torchpq``.

        Returns
        -------
        T.Tensor
            Long tensor of shape ``(n_points,)``.
        """
        return self._assign(x.T, self.centroids)[0]

    @T.no_grad()
    def fit(
        self,
        points: T.Tensor,
        max_iter: int = 100,
        tol: float = 1e-4,
        init: str = "kmeans++",
        generator: T.Generator | None = None,
    ) -> T.Tensor:
        """Fit the centroids with Lloyd's algorithm and set the class weights.

        Empty clusters are re-seeded with random points. The class weights are
        the inverse cluster frequencies on ``points``, normalised to mean one
        over the used clusters, and zero for clusters that stay empty.

        Parameters
        ----------
        points : T.Tensor
            Points of shape ``(n_points, dim)`` on the device used for fitting.
        max_iter : int, optional
            Maximum number of Lloyd iterations.
        tol : float, optional
            Stop when the relative decrease of the mean squared distance to the
            nearest centroid falls below this value.
        init : str, optional
            ``"kmeans++"`` or ``"random"`` (centroids drawn from the points).
        generator : T.Generator, optional
            Random generator on the device of ``points``, for reproducibility.

        Returns
        -------
        T.Tensor
            Cluster index of each point under the final centroids.
        """
        points = points.float()
        n_points = points.shape[0]
        if n_points < self.n_clusters:
            raise ValueError(
                f"Need at least n_clusters={self.n_clusters} points, got {n_points}"
            )

        centroids = self._init_centroids(points, init, generator)  # (K, dim)
        prev_inertia = float("inf")
        for it in range(max_iter):
            labels, sq_dist = self._assign(points, centroids.T)
            inertia = sq_dist.mean().item()

            # Move each centroid to the mean of its points
            counts = T.bincount(labels, minlength=self.n_clusters)
            sums = T.zeros_like(centroids).index_add_(0, labels, points)
            used = counts > 0
            centroids[used] = sums[used] / counts[used, None]

            # Re-seed empty clusters with random points
            n_empty = int((~used).sum())
            if n_empty:
                idx = T.randperm(n_points, device=points.device, generator=generator)
                centroids[~used] = points[idx[:n_empty]]

            rel_change = (prev_inertia - inertia) / max(prev_inertia, 1e-12)
            log.info(
                f"iter {it + 1}/{max_iter}: inertia {inertia:.6f}, "
                f"empty clusters {n_empty}"
            )
            if n_empty == 0 and 0 <= rel_change < tol:
                break
            prev_inertia = inertia

        self.centroids.copy_(centroids.T)

        # Inverse-frequency class weights under the final centroids
        labels = self.predict(points.T)
        counts = T.bincount(labels, minlength=self.n_clusters).float()
        used = counts > 0
        weights = T.zeros_like(counts)
        weights[used] = 1 / counts[used]
        weights[used] /= weights[used].mean()
        self.weights.copy_(weights)
        return labels

    def _assign(
        self, points: T.Tensor, centroids: T.Tensor
    ) -> tuple[T.Tensor, T.Tensor]:
        """Nearest centroid and squared distance for each point.

        ``points`` is ``(n_points, dim)`` and ``centroids`` is ``(dim, K)``.
        Autocast is disabled so that nearby centroids are resolved in float32
        even when called inside a bf16/fp16 mixed-precision training step.
        """
        n_points = points.shape[0]
        labels = T.empty(n_points, dtype=T.long, device=points.device)
        sq_dist = T.empty(n_points, dtype=T.float32, device=points.device)
        with T.autocast(device_type=points.device.type, enabled=False):
            centroids = centroids.float()
            c_sq = (centroids * centroids).sum(0)  # (K,)
            for start in range(0, n_points, self.chunk_size):
                x = points[start : start + self.chunk_size].float()
                # |x - c|^2 = |x|^2 - 2 x.c + |c|^2, where |x|^2 does not
                # change the argmin and is added back for the distance
                dist, idx = (c_sq - 2 * x @ centroids).min(dim=1)
                labels[start : start + len(x)] = idx
                sq_dist[start : start + len(x)] = (dist + (x * x).sum(1)).clamp_min(0)
        return labels, sq_dist

    def _init_centroids(
        self, points: T.Tensor, init: str, generator: T.Generator | None
    ) -> T.Tensor:
        """Initial centroids of shape ``(n_clusters, dim)`` drawn from ``points``."""
        n_points = points.shape[0]
        if init == "random":
            idx = T.randperm(n_points, device=points.device, generator=generator)
            return points[idx[: self.n_clusters]].clone()
        if init != "kmeans++":
            raise ValueError(f"Unknown init '{init}', use 'kmeans++' or 'random'")

        # k-means++: each new centroid is drawn with probability proportional to
        # the squared distance to the nearest centroid chosen so far. Sampling
        # uses the inverse CDF, which unlike T.multinomial has no size limit.
        centroids = T.empty(
            self.n_clusters, points.shape[1], dtype=points.dtype, device=points.device
        )
        first = T.randint(n_points, (1,), device=points.device, generator=generator)
        centroids[0] = points[first[0]]
        min_sq = ((points - centroids[0]) ** 2).sum(1)
        for k in range(1, self.n_clusters):
            cdf = min_sq.double().cumsum(0)
            u = T.rand(1, device=points.device, dtype=T.float64, generator=generator)
            idx = T.searchsorted(cdf, u * cdf[-1], right=True).clamp_max(n_points - 1)
            centroids[k] = points[idx[0]]
            min_sq = T.minimum(min_sq, ((points - centroids[k]) ** 2).sum(1))
        return centroids
