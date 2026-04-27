import torch
from utils.logger import LOGGER


class AssembledModel(torch.nn.Module):
    """An assembled model with embedding, backbone, and (potentially multiple) heads.
    If one head is provided, returns its output directly.
    If multiple heads are provided, returns a list of outputs.

    Args:
        backbone (torch.nn.Module): The backbone model.
        heads: The head module(s).
            Format of each element:
                - (head, on_particles: bool): where on_particles indicates whether to use particle-level output.
                - head (torch.nn.Module): The head module.
            If None, uses identity.
        embedding (torch.nn.Module | None): The embedding module. If None, uses identity.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        embedding: torch.nn.Module | None = None,
        heads: (
            torch.nn.Module | list[torch.nn.Module | tuple[torch.nn.Module, bool]] | None
        ) = None,
    ):
        super().__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            LOGGER.warning("No embedding module provided; use identity.")
            self.embedding = torch.nn.Identity()

        self.backbone = backbone

        if heads is None:
            self.is_single_head = True
            LOGGER.info("No head module provided; use identity.")
            heads = torch.nn.Identity()
        elif isinstance(heads, torch.nn.Module):
            self.is_single_head = True
            self.heads = heads
        else:
            self.is_single_head = False
            LOGGER.info(f"{len(heads)} head modules provided.")
            self.heads = []
            for i, head in enumerate(heads):
                if isinstance(head, tuple):
                    head_module, on_particles = head
                    if not isinstance(on_particles, bool):
                        raise ValueError(
                            f"on_particles must be a boolean. Found: {type(on_particles)}"
                        )
                    self.heads.append((head_module, on_particles))
                else:
                    LOGGER.info(f"Head {i} provided without on_particles flag; defaulting to False.")
                    self.heads.append((head, False))

    def forward(
        self,
        particles: torch.Tensor,
        jets: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | torch.Tensor]:
        """Forward pass through embedding, backbone, and head."""
        if isinstance(self.embedding, torch.nn.Identity):
            x = self.embedding(particles)
        else:
            x = self.embedding(particles, mask=mask)
        x, particles = self.backbone(x, mask=mask, jets=jets)
        if self.is_single_head:
            return x, self.heads(x)
        return x, [head(x) for head in self.heads]

    def train_backbone(self, train_embedding: bool = True):
        """Unfreeze backbone parameters and set to training mode."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()
        LOGGER.debug("Backbone set to train")

        if train_embedding:
            for param in self.embedding.parameters():
                param.requires_grad = True
            self.embedding.train()
            LOGGER.debug("Embedding set to train")
        else:
            LOGGER.warning(
                "Backbone set to train, but embedding specified to be remains frozen"
            )

    def freeze_backbone(self):
        """Freeze backbone parameters and set to eval mode."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.embedding.eval()
        LOGGER.debug("Backbone frozen")
