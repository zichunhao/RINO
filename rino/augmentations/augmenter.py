from typing import Any, Callable, TypeAlias
import torch
from utils.logger import LOGGER
from .apply import apply_matrices
from .lorentz_transformation import (
    get_random_lorentz_matrices,
    get_random_lorentz_matrices_axis,
)
from .rotation import get_random_rotation_z_matrices
from .masking import random_masking
from .smearing import smear

AugmentationFn: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]

DEFAULT_NO_SMEAR_KEYWORDS = {
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "jet_nparticles",
}


class Augmenter:
    """Class to handle various augmentations for particle physics data."""

    def __init__(
        self,
        labels_parts: list[str],
        labels_jet: list[str],
        need_part_p4: bool = False,
        need_jet_p4: bool = False,
        no_smear_keywords: list[str] | set[str] | None = None,
    ):
        """
        Initialize the augmenter.

        Args:
            labels_parts: List of particle feature names
            labels_jet: List of jet feature names
            need_part_p4: Whether particle 4-momentum is needed
            need_jet_p4: Whether jet 4-momentum is needed
            logger: Optional logger instance
        """

        if no_smear_keywords is None:
            self.no_smear_keywords = DEFAULT_NO_SMEAR_KEYWORDS
        else:
            self.no_smear_keywords = set(no_smear_keywords)
        LOGGER.info(
            f"feature keywords not involved in smearing: {self.no_smear_keywords}"
        )

        self.labels_parts = labels_parts
        self.labels_jet = labels_jet

        # Get indices for particle features
        self.need_part_p4 = need_part_p4
        self.part_energy_idx, self.part_no_smear_indices = self._process_features(
            labels_parts
        )
        if self.need_part_p4:
            LOGGER.info(f"particle energy index: {self.part_energy_idx}")

        # Get indices for jet features
        self.need_jet_p4 = need_jet_p4
        self.jet_energy_idx, self.jet_no_smear_indices = self._process_features(
            labels_jet
        )
        if self.need_jet_p4:
            LOGGER.info(f"jet energy index: {self.jet_energy_idx}")

        self.augmentation_registry: dict[str, AugmentationFn] = {
            "lorentz": self.lorentz_transform,
            "lorentz_axis": self.lorentz_transform_along_jet,
            "rot": self.rotation_z,
            "mask": self.random_masking,
            "smear": self.smear,
        }

    def _process_features(self, features: list[str]) -> tuple[int | None, list[int]]:
        """
        Get indices for special features that should not be smeared.

        Args:
            features: List of feature names

        Returns:
            tuple:
                - int | None: Index of energy component (None if not found)
                - list[int]: List of indices that should not be smeared

        Examples:
            For particle features like:
            ['tree/part_energy', 'tree/part_px', 'tree/part_py', 'tree/part_pz',
            'tree/part_isChargedHadron', 'tree/part_isNeutralHadron', ...]

            Returns: (0, [4, 5])

        Raises:
            ValueError: If energy component is not followed by px, py, pz when 4-momentum is needed
        """

        def _verify_momentum_components(features: list[str], energy_idx: int) -> bool:
            """Verify that energy is followed by px, py, pz components."""
            try:
                return all(
                    momentum in features[energy_idx + i + 1].lower()
                    for i, momentum in enumerate(["px", "py", "pz"])
                )
            except IndexError:
                return False

        idx_energy = None
        indices_no_smear = []

        # Process each feature
        for idx, feature in enumerate(features):
            feature_lower = feature.lower()

            # Check for energy component
            if "energy" in feature_lower:
                if idx_energy is None:
                    idx_energy = idx
                else:
                    LOGGER.warning(
                        "Multiple energy components found. "
                        "Using the first one for 4-momentum transformation."
                    )

                # Verify 4-momentum structure if needed
                if self.need_part_p4:
                    if not _verify_momentum_components(features, idx):
                        raise ValueError(
                            "Energy component must be followed by px, py, pz components "
                            "when 4-momentum is needed."
                        )

            # Check for categorical/boolean features
            if any(
                keyword.lower() in feature_lower for keyword in self.no_smear_keywords
            ):
                indices_no_smear.append(idx)

        return idx_energy, indices_no_smear

    def _apply_lorentz_transformation(
        self,
        particles: torch.Tensor,
        jets: torch.Tensor,
        mask: torch.Tensor,
        matrices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Lorentz transformation matrices to particles and jets."""
        if self.part_energy_idx is not None:
            particles = apply_matrices(
                x=particles,
                matrices=matrices,
                idx_start=self.part_energy_idx,
                in_place=True,
            )

        if self.jet_energy_idx is not None:
            jets = apply_matrices(
                x=jets,
                matrices=matrices,
                idx_start=self.jet_energy_idx,
                in_place=True,
            )

        return particles, jets, mask

    def lorentz_transform_along_jet(
        self,
        particles: torch.Tensor,
        jets: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Lorentz transformation along jet axis."""
        if not self.need_jet_p4:
            raise ValueError("Jet 4-momentum required for jet axis transformation")

        axis = jets[:, self.jet_energy_idx : self.jet_energy_idx + 4]
        matrices = get_random_lorentz_matrices_axis(axis=axis, **kwargs)

        return self._apply_lorentz_transformation(particles, jets, mask, matrices)

    def lorentz_transform(
        self,
        particles: torch.Tensor,
        jets: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random Lorentz transformation."""
        matrices = get_random_lorentz_matrices(
            N=len(particles), device=particles.device, dtype=particles.dtype, **kwargs
        )
        return self._apply_lorentz_transformation(particles, jets, mask, matrices)

    def rotation_z(
        self,
        particles: torch.Tensor,
        jets: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random rotation around z-axis."""
        matrices = get_random_rotation_z_matrices(
            N=len(particles), device=particles.device, dtype=particles.dtype
        )
        return self._apply_lorentz_transformation(particles, jets, mask, matrices)

    def random_masking(
        self,
        particles: torch.Tensor,
        jets: torch.Tensor,
        mask: torch.Tensor,
        mask_prob: float = 0.1,
        update_jets: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random masking to particles."""
        particles, mask = random_masking(
            particles=particles,
            mask=mask,
            mask_prob=mask_prob,
        )

        if update_jets and self.need_part_p4 and self.need_jet_p4:
            part_p4 = particles[..., self.part_energy_idx : self.part_energy_idx + 4]
            jets[:, self.jet_energy_idx : self.jet_energy_idx + 4] = part_p4.sum(dim=-2)

        return particles, jets, mask

    def smear(
        self,
        particles: torch.Tensor,
        jets: torch.Tensor,
        mask: torch.Tensor,
        smear_factor: float = 0.1,
        target: str = "particles",
        update_jets: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random smearing to particles or jets."""
        if "part" in target.lower():
            particles = smear(
                features=particles,
                smear_factor=smear_factor,
                indices_ignore=self.part_no_smear_indices,
            )
            if update_jets and self.need_part_p4 and self.need_jet_p4:
                part_p4 = particles[
                    ..., self.part_energy_idx : self.part_energy_idx + 4
                ]
                jets[:, self.jet_energy_idx : self.jet_energy_idx + 4] = part_p4.sum(
                    dim=-2
                )
        elif "jet" in target.lower():
            jets = smear(
                features=jets,
                smear_factor=smear_factor,
                indices_ignore=self.jet_no_smear_indices,
            )
        else:
            raise ValueError(
                f"Smearing target {target} not supported. "
                "Supported types: particles, jets"
            )

        return particles, jets, mask
    
    def cluster(
        self,
        particles: torch.Tensor,
        jets: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: integrate clustering augmentation
        # Maybe cluster in the upstream
        NotImplementedError

    def __call__(
        self,
        particles: torch.Tensor,
        jets: torch.Tensor,
        mask: torch.Tensor,
        config_aug: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply a sequence of augmentations based on config.
        The config should be {"type": "aug_type", "args": {"arg1": val1, ...}, ["prob": 0.5]},
        where
            - "aug_type" is the name of the augmentation
            - "args" are the extra arguments other than particles, jets, and mask,
            - "prob" is the probability of applying the augmentation (default: 1.0).
        """
        if not config_aug:
            return particles, jets, mask

        for aug_params in config_aug:
            aug_type = aug_params["type"]
            aug_args = aug_params.get("args", {})
            aug_prob = aug_params.get("prob", 1.0)
            if torch.rand(1).item() > aug_prob:
                continue

            aug_fn = self.get_augmentation_fn(aug_type)
            particles, jets, mask = aug_fn(
                particles.clone(), jets.clone(), mask.clone(), **aug_args
            )

        return particles, jets, mask

    def get_augmentation_fn(self, aug_type: str) -> AugmentationFn:
        """Get augmentation function by name."""
        return self.augmentation_registry[aug_type]
