import math
import torch
import torch.nn as nn
from utils.logger import LOGGER


def _to_polar_log(
    coords: torch.Tensor, r_scale: float = 1.0
) -> torch.Tensor:
    """Convert 2D Cartesian (dη, dφ) to polar-log coordinates.

    Output is (log(1 + r/r_scale), cos φ, sin φ).

    Rationale:
        - ``log(1 + r/r_scale)`` compresses large radii and gives finer
          resolution near (0, 0) where jet constituents are concentrated.
        - ``(cos φ, sin φ)`` is the canonical linear representation of SO(2):
          it is continuous (no ±π discontinuity), and under a rotation by θ
          the pair transforms exactly as a 2×2 rotation matrix acts on it.

    Args:
        coords: (..., 2) tensor of (dη, dφ).
        r_scale: Scale applied before the log: r → log(1 + r/r_scale).
                 Setting r_scale to the typical jet radius (e.g. 0.4–1.0)
                 keeps log_r in a numerically comfortable range.
    Returns:
        (..., 3) tensor of (log_r, cos_phi, sin_phi).
    """
    deta = coords[..., 0]
    dphi = coords[..., 1]
    r = torch.sqrt(deta**2 + dphi**2)
    phi = torch.atan2(dphi, deta)
    log_r = torch.log1p(r / r_scale)
    return torch.stack([log_r, torch.cos(phi), torch.sin(phi)], dim=-1)


class FourierFeatures(nn.Module):
    """Fixed or learnable Random Fourier Features positional encoding.

    Supports two projection modes:
        - coupled: Single (num_freqs, in_features) joint projection matrix.
        - decoupled: One (num_freqs, 1) matrix per coordinate.

    Args:
        in_features: Number of input coordinates.
        out_features: Output dimensionality (should equal d_model).
        std: Init std. Scalar for coupled; scalar or per-coord list for decoupled.
        learnable: If True, frequencies are learned. If False, fixed RFF.
        decoupled: If True, use per-coordinate frequency matrices.
        input_indices: Indices into particle feature vector to use as coords.
    """

    def __init__(
        self,
        input_indices: list[int],
        out_features: int,
        std: float | list[float] = 1.0,
        learnable: bool = False,
        decoupled: bool = False,
    ):
        super().__init__()
        self.out_features = out_features
        self.learnable = learnable
        self.decoupled = decoupled
        self.input_indices = input_indices
        in_features = len(input_indices)
        self.in_features = in_features

        num_freqs = (out_features + 1) // 2

        if decoupled:
            if isinstance(std, (int, float)):
                stds = [float(std)] * in_features
            else:
                if len(std) != in_features:
                    raise ValueError(
                        f"len(std)={len(std)} must match in_features={in_features}"
                    )
                stds = list(std)

            LOGGER.info(
                f"FourierFeatures (decoupled) in={in_features} out={out_features} "
                f"std={stds} learnable={learnable} input_indices={input_indices}"
            )
            for i, s in enumerate(stds):
                w = torch.randn(num_freqs, 1) * s
                if learnable:
                    self.register_parameter(f"weight_{i}", nn.Parameter(w))
                else:
                    self.register_buffer(f"weight_{i}", w)
        else:
            if not isinstance(std, (int, float)):
                raise ValueError("std must be a scalar in coupled mode")

            LOGGER.info(
                f"FourierFeatures (coupled) in={in_features} out={out_features} "
                f"std={std} learnable={learnable} input_indices={input_indices}"
            )
            w = torch.randn(num_freqs, in_features) * std
            if learnable:
                self.weight = nn.Parameter(w)
            else:
                self.register_buffer("weight", w)

    def _get_weight(self, i: int) -> torch.Tensor:
        if self.learnable:
            return self.get_parameter(f"weight_{i}")
        return self.get_buffer(f"weight_{i}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_features) — normalized coordinates
        Returns:
            (B, N, out_features)
        """
        if self.decoupled:
            proj = sum(
                x[..., i : i + 1] @ self._get_weight(i).T
                for i in range(self.in_features)
            )
        else:
            proj = x @ self.weight.T  # (B, N, num_freqs)

        f = 2 * math.pi * proj
        embed = torch.cat([torch.sin(f), torch.cos(f)], dim=-1)
        return embed[..., : self.out_features]


class BinnedFeatures(nn.Module):
    """ViT-style learned patch positional encoding over an N-dimensional coordinate space.

    The coordinate space is partitioned into a grid defined by per-coordinate bin specs.
    Each particle is assigned to a cell by quantizing its coordinates into a flattened
    patch index, which is looked up in a single learned embedding table.

    Args:
        out_features: Output dimensionality (should equal d_model).
        bins: Per-coordinate bin specs, each [vmin, vmax, nbins].
        input_indices: Indices into particle feature vector for each coordinate, in order.
    """

    def __init__(
        self,
        out_features: int,
        bins: list[list[float | int]],
        input_indices: list[int] | None = None,
    ):
        super().__init__()
        self.out_features = out_features
        self.bins = bins
        self.input_indices = input_indices
        self.in_features = len(bins)

        num_patches = 1
        for _, _, nbins in bins:
            num_patches *= nbins

        self.embedding = nn.Embedding(num_patches, out_features)
        nn.init.normal_(self.embedding.weight, std=0.02)

        LOGGER.info(
            f"BinnedFeatures (ViT-style) out={out_features} "
            f"bins={bins} ({num_patches} patches) "
            f"input_indices={input_indices}"
        )

    def _quantize(
        self, values: torch.Tensor, vmin: float, vmax: float, nbins: int
    ) -> torch.LongTensor:
        """Map continuous values to bin indices in [0, nbins - 1]."""
        normalized = (values.clamp(vmin, vmax) - vmin) / (vmax - vmin)
        return (normalized * (nbins - 1)).round().long().clamp(0, nbins - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_features) — coordinates
        Returns:
            (B, N, out_features)
        """
        patch_idx = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        for i, (vmin, vmax, nbins) in enumerate(self.bins):
            patch_idx = patch_idx * nbins + self._quantize(x[..., i], vmin, vmax, nbins)
        return self.embedding(patch_idx)


class PolarHarmonicFeatures(nn.Module):
    """Continuous, exactly cyclic positional encoding in polar-log space.

    Separable encoding:

    - **Angular** — circular harmonics ``[cos(nφ), sin(nφ)]`` for
      ``n = 1 … n_ang``.  These are the eigenfunctions of rotation and are
      exactly 2π-periodic: φ = π and φ = −π map to the same point, and
      particles that are close in angle always get close encodings.

    - **Radial** — 1-D sinusoidal basis (RFF) on ``log(1 + r/r_scale)``.
      Smooth and monotone in r; log-compression gives finer resolution near
      the jet core where most particles live.

    Total dimension = ``2 * n_ang  +  2 * n_radial_freqs = out_features``.

    Compared with ``PolarFourierFeatures`` (RFF on the 3-D polar-log vector),
    this encoding has explicit frequency control per angular order and an exact
    Fourier series structure in φ.

    Args:
        input_indices: Indices of (dη, dφ) in the full particle feature vector.
                       Must have length 2.
        out_features: Output dimensionality (should equal d_model).
        n_ang: Number of angular harmonics (orders 1 … n_ang).
               Default: ``out_features // 4``, giving an equal radial/angular
               dimension split.
        r_scale: Radial scale: r → log(1 + r / r_scale).
        std: Init std for the radial RFF weight.
        learnable: If True, radial frequencies are learned.
    """

    def __init__(
        self,
        input_indices: list[int],
        out_features: int,
        n_ang: int | None = None,
        r_scale: float = 1.0,
        std: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()
        if len(input_indices) != 2:
            raise ValueError(
                f"PolarHarmonicFeatures requires exactly 2 input_indices (dη, dφ), "
                f"got {len(input_indices)}"
            )
        self.input_indices = input_indices
        self.r_scale = r_scale
        self.learnable = learnable

        n_ang = n_ang if n_ang is not None else out_features // 4
        n_ang_dims = 2 * n_ang
        n_rad_dims = out_features - n_ang_dims
        if n_rad_dims < 2:
            raise ValueError(
                f"n_ang={n_ang} leaves only {n_rad_dims} dims for the radial part; "
                f"reduce n_ang or increase out_features."
            )
        n_radial_freqs = n_rad_dims // 2  # sin + cos per frequency

        self.out_features = out_features
        self.n_ang = n_ang
        self.n_ang_dims = n_ang_dims
        self.n_radial_freqs = n_radial_freqs

        # Angular orders 1, 2, …, n_ang  (fixed, non-learnable by design)
        self.register_buffer("orders", torch.arange(1, n_ang + 1, dtype=torch.float32))

        # 1-D radial RFF on log_r
        w = torch.randn(n_radial_freqs, 1) * std
        if learnable:
            self.radial_weight = nn.Parameter(w)
        else:
            self.register_buffer("radial_weight", w)

        LOGGER.info(
            f"PolarHarmonicFeatures out={out_features} "
            f"n_ang={n_ang} (dims={n_ang_dims}) "
            f"n_radial={n_radial_freqs} (dims={n_rad_dims}) "
            f"r_scale={r_scale} learnable={learnable} "
            f"input_indices={input_indices}"
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2) — already-extracted (dη, dφ) coordinates.
        Returns:
            (B, N, out_features)
        """
        deta = coords[..., 0]
        dphi = coords[..., 1]
        r   = torch.sqrt(deta**2 + dphi**2)
        phi = torch.atan2(dphi, deta)          # (B, N),  range (−π, π]
        log_r = torch.log1p(r / self.r_scale)  # (B, N)

        # ── angular: [cos(nφ), sin(nφ)] for n = 1…n_ang ──────────────────
        # n_phi: (B, N, n_ang)
        n_phi = phi.unsqueeze(-1) * self.orders
        ang = torch.cat([torch.cos(n_phi), torch.sin(n_phi)], dim=-1)  # (B, N, 2*n_ang)

        # ── radial: 1-D RFF on log_r ──────────────────────────────────────
        # log_r: (B, N, 1) @ (n_radial_freqs, 1).T → (B, N, n_radial_freqs)
        rad_proj = 2 * math.pi * (log_r.unsqueeze(-1) @ self.radial_weight.T)
        rad = torch.cat([torch.sin(rad_proj), torch.cos(rad_proj)], dim=-1)  # (B, N, 2*n_radial)

        # ── concatenate and truncate to out_features ──────────────────────
        return torch.cat([ang, rad], dim=-1)[..., : self.out_features]


class PolarFourierFeatures(nn.Module):
    """RFF positional encoding in polar-log coordinate space.

    Receives already-extracted (dη, dφ) coordinates (shape ``(B, N, 2)``),
    converts them internally to ``(log(1+r/r_scale), cos φ, sin φ)``, then
    applies Random Fourier Features on this 3-dimensional representation.

    This addresses two shortcomings of Cartesian RFF:

    1. **SO(2) symmetry** — ``(cos φ, sin φ)`` is the standard linear
       representation of SO(2), so the encoding does not privilege any
       particular axis in the (η, φ) plane.
    2. **Density near origin** — ``log(1+r/r_scale)`` compresses large
       radii and allocates more frequency resolution near (0, 0) where
       most jet constituents live.

    Args:
        input_indices: Indices of (dη, dφ) in the full particle feature vector.
                       Must have length 2.
        out_features: Output dimensionality (should equal d_model).
        r_scale: Radial scale before log: r → log(1 + r / r_scale).
        std: Init std for the RFF weight matrix.
        learnable: If True, frequencies are learned; otherwise fixed RFF.
        decoupled: If True, use separate frequency matrices per coordinate.
    """

    def __init__(
        self,
        input_indices: list[int],
        out_features: int,
        r_scale: float = 1.0,
        std: float | list[float] = 1.0,
        learnable: bool = False,
        decoupled: bool = False,
    ):
        super().__init__()
        if len(input_indices) != 2:
            raise ValueError(
                f"PolarFourierFeatures requires exactly 2 input_indices (dη, dφ), "
                f"got {len(input_indices)}"
            )
        self.input_indices = input_indices
        self.r_scale = r_scale

        # Internal FourierFeatures operates on 3D polar-log coords:
        # (log_r, cos_phi, sin_phi).  We pass dummy input_indices=[0,1,2]
        # because coordinate extraction is done here before calling it.
        self._fourier = FourierFeatures(
            input_indices=[0, 1, 2],
            out_features=out_features,
            std=std,
            learnable=learnable,
            decoupled=decoupled,
        )

        LOGGER.info(
            f"PolarFourierFeatures out={out_features} r_scale={r_scale} "
            f"std={std} learnable={learnable} decoupled={decoupled} "
            f"input_indices={input_indices}"
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2) — already-extracted (dη, dφ) coordinates.
        Returns:
            (B, N, out_features)
        """
        polar = _to_polar_log(coords, self.r_scale)  # (B, N, 3)
        return self._fourier(polar)


class PolarBinnedFeatures(nn.Module):
    """Binned positional encoding in polar-log coordinate space.

    Receives already-extracted (dη, dφ) coordinates (shape ``(B, N, 2)``),
    converts them to ``(log(1+r/r_scale), φ)`` and bins each dimension
    independently to form a ring × sector patch index for an embedding lookup.

    Compared with Cartesian binning this:

    1. **Respects SO(2) structure** — patches are annular sectors rather than
       Cartesian squares, so a rotation maps sector patches to sector patches.
    2. **Concentrates resolution near origin** — equal-width bins in log-space
       correspond to geometrically-spaced rings: the innermost ring is narrow
       (high spatial resolution) and outer rings grow progressively wider.

    Args:
        input_indices: Indices of (dη, dφ) in the full particle feature vector.
                       Must have length 2.
        out_features: Output dimensionality (should equal d_model).
        n_r_bins: Number of radial (ring) bins.
        n_phi_bins: Number of angular (sector) bins.
        r_max: Maximum radius; particles beyond this are clamped to the
               outermost ring.  Set to a value that covers the jet cone.
        r_scale: Radial scale: r → log(1 + r / r_scale).
    """

    def __init__(
        self,
        input_indices: list[int],
        out_features: int,
        n_r_bins: int = 32,
        n_phi_bins: int = 64,
        r_max: float = 3.0,
        r_scale: float = 1.0,
    ):
        super().__init__()
        if len(input_indices) != 2:
            raise ValueError(
                f"PolarBinnedFeatures requires exactly 2 input_indices (dη, dφ), "
                f"got {len(input_indices)}"
            )
        self.input_indices = input_indices
        self.r_scale = r_scale

        log_r_max = math.log1p(r_max / r_scale)
        num_patches = n_r_bins * n_phi_bins

        # Reuse BinnedFeatures with 2D (log_r, phi) bins
        self._binned = BinnedFeatures(
            out_features=out_features,
            bins=[[0.0, log_r_max, n_r_bins], [-math.pi, math.pi, n_phi_bins]],
        )

        LOGGER.info(
            f"PolarBinnedFeatures out={out_features} "
            f"n_r={n_r_bins} n_phi={n_phi_bins} r_max={r_max} r_scale={r_scale} "
            f"({num_patches} patches) input_indices={input_indices}"
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2) — already-extracted (dη, dφ) coordinates.
        Returns:
            (B, N, out_features)
        """
        deta = coords[..., 0]
        dphi = coords[..., 1]
        r = torch.sqrt(deta**2 + dphi**2)
        phi = torch.atan2(dphi, deta)
        log_r = torch.log1p(r / self.r_scale)
        polar = torch.stack([log_r, phi], dim=-1)  # (B, N, 2)
        return self._binned(polar)


class PositionalEncoding(nn.Module):
    """Unified positional encoding wrapper for particle-level coordinates.

    Supports five modes, selected at construction time:

    - ``"fourier"``: Fixed or learnable Random Fourier Features on raw
      Cartesian (dη, dφ) coordinates.
    - ``"binned"``: Discretized lookup-table embeddings on a Cartesian grid.
    - ``"polar_fourier"``: RFF applied to polar-log coordinates
      ``(log(1+r/r_scale), cos φ, sin φ)``.  Continuous, approximately cyclic.
    - ``"polar_binned"``: Binned embeddings on a ring × sector grid in
      log-radial space.
    - ``"polar_harmonic"``: Separable continuous encoding — exact circular
      harmonics ``[cos(nφ), sin(nφ)]`` for the angular part and 1-D RFF on
      ``log(1+r/r_scale)`` for the radial part.  The angular basis is exactly
      2π-periodic by construction; adjacent points always get similar
      encodings.

    In all modes, ``input_indices`` selects which dimensions of the full
    particle feature vector to treat as coordinates. The encoding is always
    applied in ``d_model`` space (post particle_embedding), summed additively.

    Args:
        mode: One of ``"fourier"``, ``"binned"``, ``"polar_fourier"``,
              ``"polar_binned"``.
        input_indices: Indices into the particle feature vector for coordinates.
                       Polar modes require exactly 2 indices (dη, dφ).
        out_features: Must equal backbone ``d_model``.
        kwargs: Forwarded to the underlying encoder class.
    """

    def __init__(
        self,
        mode: str,
        input_indices: list[int],
        out_features: int,
        kwargs: dict | None = None,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.input_indices = input_indices
        kwargs = kwargs or {}

        if self.mode == "fourier":
            self.encoder = FourierFeatures(
                input_indices=input_indices,
                out_features=out_features,
                **kwargs,
            )
        elif self.mode == "binned":
            self.encoder = BinnedFeatures(
                out_features=out_features,
                input_indices=input_indices,
                **kwargs,
            )
        elif self.mode == "polar_fourier":
            self.encoder = PolarFourierFeatures(
                input_indices=input_indices,
                out_features=out_features,
                **kwargs,
            )
        elif self.mode == "polar_binned":
            self.encoder = PolarBinnedFeatures(
                input_indices=input_indices,
                out_features=out_features,
                **kwargs,
            )
        elif self.mode == "polar_harmonic":
            self.encoder = PolarHarmonicFeatures(
                input_indices=input_indices,
                out_features=out_features,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown PE mode: {mode!r}. "
                f"Choose 'fourier', 'binned', 'polar_fourier', 'polar_binned', "
                f"or 'polar_harmonic'."
            )

        LOGGER.info(
            f"PositionalEncoding mode={self.mode} input_indices={input_indices}"
        )

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """Extract coords from particles and compute positional embedding.

        Args:
            particles: (B, N, part_dim) — full unmasked particle feature tensor
        Returns:
            (B, N, out_features)
        """
        coords = particles[..., self.input_indices]  # (B, N, len(input_indices))
        return self.encoder(coords)
