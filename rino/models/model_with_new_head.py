import torch
from utils.logger import LOGGER

class ModelWithNewHead(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
        LOGGER.info(f"Initializing ModelWithNewHead with backbone: {self.backbone} and head: {self.head}")
        if hasattr(self.backbone, 'head'):
            LOGGER.info("Backbone has a head, freezing its parameters.")
            for param in self.backbone.head.parameters():
                param.requires_grad = False

    def forward(
        self, 
        particles: torch.Tensor, 
        jets: torch.Tensor, 
        mask: torch.Tensor | None = None,
        include_dino_head: bool = False,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if include_dino_head:
            proj_dino, rep = self.backbone(particles=particles, jets=jets, mask=mask, use_head=True)
            proj_class = self.head(rep)
            return (proj_class, proj_dino), rep
        else:
            rep = self.backbone(particles=particles, jets=jets, mask=mask, use_head=False)
            proj_class = self.head(rep)
            return proj_class, rep
    
    def train_backbone(self, include_backbone_head: bool = False):
        """Unfreeze backbone parameters and set to training mode."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()
        
        # Re-freeze backbone head if it exists and not explicitly included
        if not include_backbone_head and hasattr(self.backbone, 'head'):
            for param in self.backbone.head.parameters():
                param.requires_grad = False
        
        LOGGER.debug(f"Backbone unfrozen (include_backbone_head={include_backbone_head})")

    def freeze_backbone(self):
        """Freeze backbone parameters and set to eval mode."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        LOGGER.debug("Backbone frozen")
        
    def freeze_embedding(self):
        """Freeze the backbone's embedding parameters."""
        if hasattr(self.backbone, "freeze_embedding"):
            self.backbone.freeze_embedding()
        else:
            raise AttributeError("The backbone does not have a 'freeze_embedding' method.")
