import math
import torch
import torch.nn as nn
from utils.logger import LOGGER


def _to_polar_log(coords: torch.Tensor, r_scale: float = 1.0) -> torch.Tensor:
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

    Optionally applies a signed-log transform per coordinate before binning:

        x → sign(x) · log(1 + |x| / log_scale)

    This compresses large magnitudes and concentrates bin resolution near zero,
    matching the density of jet constituent coordinates (dη, dφ, etc.).
    For non-negative coordinates the sign flip has no effect.  ``vmin``/``vmax``
    in ``bins`` are always specified in **original coordinate space**; when
    ``log_bins`` is enabled the bounds are converted to log space internally.

    Args:
        out_features: Output dimensionality (should equal d_model).
        bins: Per-coordinate bin specs, each [vmin, vmax, nbins], always in
              original coordinate space.
        input_indices: Indices into particle feature vector for each coordinate, in order.
        log_bins: Per-coordinate flag enabling the signed-log transform.  A single
                  bool applies the same setting to all coordinates.  Default: False.
        log_scale: Scale applied before the log: |x| → log(1 + |x| / log_scale).
                   A single float applies to all log-binned coordinates; a list
                   gives per-coordinate scales.  Default: 1.0.
    """

    def __init__(
        self,
        out_features: int,
        bins: list[list[float | int]],
        input_indices: list[int] | None = None,
        log_bins: bool | list[bool] = False,
        log_scale: float | list[float] = 1.0,
    ):
        super().__init__()
        self.out_features = out_features
        self.bins = bins
        self.input_indices = input_indices
        self.in_features = len(bins)

        n = self.in_features
        if isinstance(log_bins, bool):
            self.log_bins = [log_bins] * n
        else:
            if len(log_bins) != n:
                raise ValueError(
                    f"len(log_bins)={len(log_bins)} must match number of bin specs={n}"
                )
            self.log_bins = list(log_bins)

        if isinstance(log_scale, (int, float)):
            self.log_scale = [float(log_scale)] * n
        else:
            if len(log_scale) != n:
                raise ValueError(
                    f"len(log_scale)={len(log_scale)} must match number of bin specs={n}"
                )
            self.log_scale = list(log_scale)

        # Pre-compute effective (vmin, vmax) in the space seen by _quantize.
        # For log-binned coords, convert original-space bounds to log space once.
        self._effective_bins = []
        for i, (vmin, vmax, nbins) in enumerate(bins):
            if self.log_bins[i]:
                s = self.log_scale[i]
                vmin_eff = math.copysign(math.log1p(abs(vmin) / s), vmin)
                vmax_eff = math.copysign(math.log1p(abs(vmax) / s), vmax)
            else:
                vmin_eff, vmax_eff = vmin, vmax
            self._effective_bins.append((vmin_eff, vmax_eff, nbins))

        num_patches = 1
        for _, _, nbins in bins:
            num_patches *= nbins

        self.embedding = nn.Embedding(num_patches, out_features)
        nn.init.normal_(self.embedding.weight, std=0.02)

        LOGGER.info(
            f"BinnedFeatures (ViT-style) out={out_features} "
            f"bins={bins} ({num_patches} patches) "
            f"log_bins={self.log_bins} log_scale={self.log_scale} "
            f"effective_bins={self._effective_bins} "
            f"input_indices={input_indices}"
        )

    def _quantize(
        self, values: torch.Tensor, vmin: float, vmax: float, nbins: int
    ) -> torch.LongTensor:
        """Map continuous values to bin indices in [0, nbins - 1]."""
        normalized = (values.clamp(vmin, vmax) - vmin) / (vmax - vmin)
        return (normalized * (nbins - 1)).round().long().clamp(0, nbins - 1)

    def _signed_log(self, values: torch.Tensor, scale: float) -> torch.Tensor:
        """Apply sign(x) * log1p(|x| / scale) — signed-log compression."""
        return values.sign() * torch.log1p(values.abs() / scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_features) — coordinates
        Returns:
            (B, N, out_features)
        """
        patch_idx = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        for i, (vmin_eff, vmax_eff, nbins) in enumerate(self._effective_bins):
            coord = x[..., i]
            if self.log_bins[i]:
                coord = self._signed_log(coord, self.log_scale[i])
            patch_idx = patch_idx * nbins + self._quantize(
                coord, vmin_eff, vmax_eff, nbins
            )
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
                f"PolarHarmonicFeatures requires exactly 2 input_indices (x, y), "
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
        n_radial_freqs = (
            n_rad_dims + 1
        ) // 2  # sin + cos per frequency; ceil so output never falls short

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
        r = torch.sqrt(deta**2 + dphi**2)
        phi = torch.atan2(dphi, deta)  # (B, N),  range (−π, π]
        log_r = torch.log1p(r / self.r_scale)  # (B, N)

        # ── angular: [cos(nφ), sin(nφ)] for n = 1…n_ang ──────────────────
        # n_phi: (B, N, n_ang)
        n_phi = phi.unsqueeze(-1) * self.orders
        ang = torch.cat([torch.cos(n_phi), torch.sin(n_phi)], dim=-1)  # (B, N, 2*n_ang)

        # ── radial: 1-D RFF on log_r ──────────────────────────────────────
        # log_r: (B, N, 1) @ (n_radial_freqs, 1).T → (B, N, n_radial_freqs)
        rad_proj = 2 * math.pi * (log_r.unsqueeze(-1) @ self.radial_weight.T)
        rad = torch.cat(
            [torch.cos(rad_proj), torch.sin(rad_proj)], dim=-1
        )  # (B, N, 2*n_radial)

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


class RelativePositionalBias(nn.Module):
    """Pairwise ΔR attention bias for relative positional encoding.

    For each pair (i, j) computes the relative vector

        [Δx1_ij, …, ΔxC_ij,  (mag_ij,)  ΔR_ij]

    where ``Δxk_ij = xk_i − xk_j``, ``ΔR_ij = ‖Δx_ij‖``, and the optional
    ``mag_ij`` block is either ``|Δx_ij|`` or ``Δx_ij²`` depending on
    ``magnitude``.  The vector is projected to a per-head scalar bias:

        attn_logits[b, h, i, j] += bias[b, h, i, j]

    The signed differences make the bias asymmetric (bias[i,j] ≠ bias[j,i]).
    Including magnitude features alongside the signed differences lets a
    **linear** projection express piecewise-linear functions of each coordinate
    (different slopes for positive and negative separations) — equivalent to
    a minimal ReLU nonlinearity — without requiring an MLP hidden layer.

    The output projection is zero-initialised so training starts with no
    positional bias and learns it from data.

    Args:
        nhead: Number of attention heads.
        input_indices: Indices of coordinates in the particle feature vector.
                       Any length ≥ 1.
        r_scale: Optional scale applied to all inputs before the projection.
                 Divides the pairwise differences and enters the log as
                 ``log(ΔR / r_scale)``.  With a linear layer this is strictly
                 redundant but helps MLP conditioning.  ``None`` disables it.
        hidden_dim: Hidden dimension for a single intermediate GELU layer.
                    ``None`` → plain linear map (no hidden layer).
        log_dr: If ``True``, append ``log(ΔR / r_scale)`` instead of
                ``ΔR / r_scale``.
        magnitude: Whether to append per-coordinate magnitude features
                   alongside the signed differences.
                   ``None``  — signed diffs only       (input dim = C + 1)
                   ``"abs"`` — append sqrt(Δxk²+ε)    (input dim = 2C + 1)
                   ``"sq"``  — append Δxk²             (input dim = 2C + 1)
                   With a linear projection, ``"abs"`` gives piecewise-linear
                   behaviour (different slopes for each sign of Δxk).  Smooth
                   abs ``sqrt(x²+ε)`` is used instead of ``|x|`` to keep
                   gradients well-defined at coincident particle coordinates.
                   ``"sq"`` gives even-function sensitivity to magnitude while
                   keeping the signed term for asymmetry.
    """

    def __init__(
        self,
        nhead: int,
        input_indices: list[int],
        r_scale: float | None = None,
        hidden_dim: int | None = None,
        log_dr: bool = True,
        magnitude: str | None = None,
    ):
        super().__init__()
        if len(input_indices) < 1:
            raise ValueError("input_indices must be non-empty")
        if magnitude is not None and magnitude not in ("abs", "sq"):
            raise ValueError(
                f"magnitude must be 'abs', 'sq', or None, got {magnitude!r}"
            )
        self.nhead = nhead
        self.input_indices = input_indices
        self.r_scale = r_scale
        self.log_dr = log_dr
        self.magnitude = magnitude

        C = len(input_indices)
        n_in = C + (C if magnitude is not None else 0) + 1  # diffs (+ mags) + ΔR

        if hidden_dim is None:
            self.proj = nn.Linear(n_in, nhead, bias=False)
            nn.init.zeros_(self.proj.weight)
        else:
            self.proj = nn.Sequential(
                nn.Linear(n_in, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, nhead, bias=False),
            )
            nn.init.zeros_(self.proj[-1].weight)

        LOGGER.info(
            f"RelativePositionalBias nhead={nhead} input_indices={input_indices} "
            f"n_in={n_in} r_scale={r_scale} hidden_dim={hidden_dim} "
            f"log_dr={log_dr} magnitude={magnitude}"
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, C) — already-extracted coordinates,
                    C = len(input_indices).
        Returns:
            (B, nhead, N, N) — per-head attention bias to add to logits.
        """
        # Pairwise differences: (B, N, N, C)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        # ΔR: (B, N, N, 1); eps guards the diagonal (i == j) from zero-grad
        dR = torch.sqrt((diff**2).sum(dim=-1, keepdim=True) + 1e-8)

        if self.r_scale is not None:
            diff = diff / self.r_scale
            dR = dR / self.r_scale

        # log1p(dR) is bounded below at 0 (contact) and grows slowly for large ΔR.
        # Plain log(dR) diverges to -∞ at contact, which can destabilize training.
        r_feature = torch.log1p(dR) if self.log_dr else dR

        parts = [diff]
        if self.magnitude == "abs":
            # Smooth abs: sqrt(x² + ε) ≈ |x| for |x| >> sqrt(ε), differentiable at 0.
            # Plain abs has a subgradient of 0 at x=0 (PyTorch convention), which
            # silently kills gradient flow for coincident particles (same η or φ cell).
            parts.append(torch.sqrt(diff**2 + 1e-8))
        elif self.magnitude == "sq":
            parts.append(diff**2)
        parts.append(r_feature)

        rel = torch.cat(parts, dim=-1)  # (B, N, N, n_in)
        # Project to per-head bias: (B, N, N, nhead) → (B, nhead, N, N)
        bias = self.proj(rel).permute(0, 3, 1, 2)
        # Zero diagonal: at i==j, diff=0 and dR=sqrt(eps) — the eps-driven
        # log value is a numerical artifact, not a meaningful self-distance.
        # Self-attention needs no positional bias.
        bias.diagonal(dim1=-2, dim2=-1).zero_()
        return bias


class ScaleConditioning(nn.Module):
    """Post-backbone scale conditioning on the pooled jet representation.

    Injects the RG scale (number of subjets or particles) into the CLS token
    output via an additive MLP: ``rep += MLP(log(nprongs))``.

    The scale is represented as ``log(nprongs)`` so that:
    - Subjet levels 1–16 and "ALL" (~33 median, up to ~150) are spread evenly.
    - The encoding respects the natural ordering of the RG flow without
      privileging any particular discrete level.

    The output projection is zero-initialised so the module is a no-op at the
    start of training and the scale bias is learned from scratch.

    When ``nprongs`` is not provided to the backbone forward, it is inferred
    from ``mask.sum(dim=-1)`` (particle count of the current view).

    Args:
        d_model: Backbone hidden dimension (must match the pooled rep dim).
        hidden_dim: Hidden layer width. Default 64.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        LOGGER.info(f"ScaleConditioning d_model={d_model} hidden_dim={hidden_dim}")

    def forward(self, log_n: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_n: (B,) — log(nprongs) per sample.
        Returns:
            (B, d_model) — additive bias for the pooled representation.
        """
        return self.mlp(log_n.unsqueeze(-1))


class ScaleProjection(nn.Module):
    """Project out scale-dependent subspace from jet representations.

    Instead of *adding* scale information (like ``ScaleConditioning``), this
    module *removes* it: it learns a scale-dependent direction (or subspace)
    in representation space and projects the backbone output orthogonally to
    that direction.  The result is a representation that is explicitly purged
    of scale-dependent content — a geometric construction analogous to gauge
    fixing or orbit projection.

    For rank-1 the operation is::

        s_hat = normalize(MLP(log n))
        r_out = r - gate * (r · s_hat) s_hat

    where ``gate`` is bounded to ``[0, gate_max]`` via
    ``gate_max * sigmoid(gate_raw)``.  ``gate_raw`` is initialised to 0, so
    ``gate`` starts at ``gate_max / 2`` — a small but non-zero projection from
    the first step.  The sigmoid ceiling prevents the gate from growing large
    enough to remove discriminative (non-scale) content, which caused
    catastrophic collapse in the unconstrained version.

    For rank > 1, ``MLP`` outputs ``rank`` vectors that are orthogonalised via
    QR decomposition before projecting.

    Args:
        d_model: Backbone representation dimension.
        hidden_dim: MLP hidden width.  Default 64.
        rank: Dimensionality of the scale subspace to project out.  Default 1.
        gate_max: Upper bound on the gate value.  Default 0.5.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        rank: int = 1,
        gate_max: float = 0.5,
    ):
        super().__init__()
        self.rank = rank
        self.gate_max = gate_max
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model * rank),
        )
        # Raw gate parameter passed through sigmoid, then scaled by gate_max.
        # gate_raw=0 → gate=gate_max/2 at init, so projection starts half-open
        # but is bounded to [0, gate_max] throughout training.
        # gate_max=0.5 prevents the projection from ever removing more than half
        # of the scale component in one step, avoiding the catastrophic collapse
        # seen with an unconstrained gate (epoch-5 collapse in gwide-pbin-scproj).
        self.gate_raw = nn.Parameter(torch.zeros(1))
        LOGGER.info(
            f"ScaleProjection d_model={d_model} hidden_dim={hidden_dim} "
            f"rank={rank} gate_max={gate_max}"
        )

    def forward(self, rep: torch.Tensor, log_n: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rep: (B, d_model) — pooled jet representation from backbone.
            log_n: (B,) — log(nprongs) per sample.
        Returns:
            (B, d_model) — representation with scale subspace projected out.
        """
        s = self.mlp(log_n.unsqueeze(-1))  # (B, d_model * rank)

        if self.rank == 1:
            s_hat = torch.nn.functional.normalize(s, dim=-1)  # unit direction
            coeff = (rep * s_hat).sum(dim=-1, keepdim=True)  # (B, 1)
            proj = coeff * s_hat  # (B, d_model)
        else:
            # Multi-rank: orthogonalise via QR then project
            s = s.view(rep.size(0), self.rank, -1)  # (B, rank, d_model)
            q, _ = torch.linalg.qr(s.transpose(-1, -2))  # (B, d_model, rank)
            proj = (q @ (q.mT @ rep.unsqueeze(-1))).squeeze(-1)  # (B, d_model)

        gate = self.gate_max * torch.sigmoid(self.gate_raw)
        return rep - gate * proj


class RankEmbedding(nn.Module):
    """Positional encoding based on pT rank order.

    Assigns each particle an integer rank (0 = hardest) and looks it up in a
    learnable embedding table.  Encodes energy hierarchy rather than spatial
    position, complementary to coordinate-based encodings.

    If ``pt_sorted=True`` (default), particles are assumed to already be sorted
    by pT descending; rank equals position index and no argsort is needed.
    Padded particles naturally receive higher rank indices (they trail all valid
    particles in the sorted sequence) and are ignored by the transformer's
    attention padding mask.

    If ``pt_sorted=False``, ranks are computed via argsort on the feature at
    ``pt_index``.  Pass the valid-particle boolean mask (True = valid) to
    ``forward`` so that padding positions are pushed to the end before sorting.

    Args:
        out_features: Output dimensionality (must equal d_model).
        max_seq_len: Size of the embedding table (>= max particles per jet).
        pt_index: Feature index of pT/energy in the particle vector.
                  Only used when ``pt_sorted=False``.
        pt_sorted: If True, use position index as rank (no argsort needed).
    """

    def __init__(
        self,
        out_features: int,
        max_seq_len: int,
        pt_index: int | None = None,
        pt_sorted: bool = False,
    ):
        super().__init__()
        self.out_features = out_features
        self.max_seq_len = max_seq_len
        self.pt_index = pt_index
        self.pt_sorted = pt_sorted
        if self.pt_sorted:
            LOGGER.info(
                "RankEmbedding initialized with pt_sorted=True; "
                "particles are assumed to be pre-sorted by pT and pt_index is ignored."
            )
        else:
            if pt_index is None:
                raise ValueError("pt_index must be provided when pt_sorted=False")
        self.embedding = nn.Embedding(max_seq_len, out_features)

        LOGGER.info(
            f"RankEmbedding out={out_features} max_seq_len={max_seq_len} "
            f"pt_sorted={pt_sorted} pt_index={pt_index}"
        )

    def forward(
        self, particles: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            particles: (B, N, part_dim)
            mask: (B, N) bool, True = valid particle. Used only when
                  ``pt_sorted=False`` to push padding positions to the end
                  before computing ranks.
        Returns:
            (B, N, out_features)
        """
        B, N, _ = particles.shape

        if self.pt_sorted:
            ranks = torch.arange(N, device=particles.device).unsqueeze(0).expand(B, -1)
        else:
            pt = particles[..., self.pt_index].clone()  # (B, N)
            if mask is not None:
                # Push padded positions to the end before sorting
                pt = pt.masked_fill(~mask, float("-inf"))
            order = torch.argsort(pt, dim=1, descending=True)  # (B, N)
            ranks = torch.argsort(order, dim=1)  # rank of each position

        return self.embedding(ranks)


class PositionalEncoding(nn.Module):
    """Unified positional encoding wrapper for particle-level coordinates.

    Supports six modes, selected at construction time:

    - ``"fourier"``: Fixed or learnable Random Fourier Features on raw
      Cartesian (dη, dφ) coordinates.
    - ``"binned"``: Discretized lookup-table embeddings on a Cartesian grid.
      Supports optional per-coordinate signed-log binning via ``log_bins`` and
      ``log_scale`` kwargs (see ``BinnedFeatures``).
    - ``"polar_fourier"``: RFF applied to polar-log coordinates
      ``(log(1+r/r_scale), cos φ, sin φ)``.  Continuous, approximately cyclic.
    - ``"polar_binned"``: Binned embeddings on a ring × sector grid in
      log-radial space.
    - ``"polar_harmonic"``: Separable continuous encoding — exact circular
      harmonics ``[cos(nφ), sin(nφ)]`` for the angular part and 1-D RFF on
      ``log(1+r/r_scale)`` for the radial part.  The angular basis is exactly
      2π-periodic by construction; adjacent points always get similar
      encodings.
    - ``"rank"``: Learnable embedding indexed by pT rank (0 = hardest
      particle).  Encodes energy hierarchy; does not use spatial coordinates.
      Requires ``max_seq_len`` in ``kwargs``.  ``input_indices`` is ignored.

    In coordinate modes, ``input_indices`` selects which dimensions of the
    full particle feature vector to treat as coordinates. The encoding is
    always applied in ``d_model`` space (post particle_embedding), summed
    additively.

    Args:
        mode: One of ``"fourier"``, ``"binned"``, ``"polar_fourier"``,
              ``"polar_binned"``, ``"polar_harmonic"``, ``"rank"``.
        input_indices: Indices into the particle feature vector for
                       coordinates. Polar modes require exactly 2 indices
                       (dη, dφ). Ignored in ``"rank"`` mode.
        out_features: Must equal backbone ``d_model``.
        kwargs: Forwarded to the underlying encoder class.
    """

    def __init__(
        self,
        mode: str,
        input_indices: list[int] | None,
        out_features: int,
        kwargs: dict | None = None,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.input_indices = input_indices or []
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
        elif self.mode == "rank":
            self.encoder = RankEmbedding(
                out_features=out_features,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown PE mode: {mode!r}. "
                f"Choose 'fourier', 'binned', 'polar_fourier', 'polar_binned', "
                f"'polar_harmonic', or 'rank'."
            )

        LOGGER.info(
            f"PositionalEncoding mode={self.mode} input_indices={input_indices}"
        )

    def forward(
        self, particles: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract coords from particles and compute positional embedding.

        Args:
            particles: (B, N, part_dim) — full unmasked particle feature tensor
            mask: (B, N) bool, True = valid particle. Forwarded to
                  ``RankEmbedding`` when ``mode="rank"`` and
                  ``pt_sorted=False``; ignored in all other modes.
        Returns:
            (B, N, out_features)
        """
        if self.mode == "rank":
            return self.encoder(particles, mask=mask)
        coords = particles[..., self.input_indices]  # (B, N, len(input_indices))
        return self.encoder(coords)
