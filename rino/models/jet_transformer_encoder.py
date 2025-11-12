import torch
import torch.nn.functional as F
from torch import nn
from typing import Literal
from utils.logger import LOGGER
from .head import DINOHead

EPS = 1e-6


class JetTransformerEncoder(nn.Module):
    """
    A transformer encoder for jets, based on the vanilla transformer encoder.

    Features:
    - Three pooling strategies:
        - 'mean': Mean pooling over all particle tokens
        - 'cls_token': Uses a learnable class token for pooling
        - 'cls_jet': Uses jet features to initialize the class token (default)

    Args:
        part_dim: Dimension of the input particle-level data
        proj_dim: Dimension of the output representation
        d_model: Internal dimension of the transformer model
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        jet_dim: Dimension of the input jet-level data (optional, required for 'cls_jet')
        batch_first: Use batch-first tensor format
        dim_feedforward: Dimension of transformer's feed-forward network
        pooling: Pooling strategy ('mean', 'cls_token', or 'cls_jet')
        d_head_hidden: Hidden dimension for projection head
        head_l2_norm: Apply L2 normalization to the projection head's output before the last layer
        head_weight_norm: Apply weight normalization to the last layer of the projection head
        **kwargs: Additional arguments for torch.nn.TransformerEncoderLayer
    """

    def __init__(
        self,
        part_dim: int,
        proj_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        activation: str = "gelu",
        jet_dim: int | None = None,
        batch_first: bool = True,
        dim_feedforward: int = 2048,
        pooling: Literal["mean", "cls_token", "cls_jet"] = "cls_jet",
        d_head_hidden: int = 2048,
        num_head_layers: int = 3,
        head_l2_norm: bool = True,
        head_weight_norm: bool = True,
        xavier_init: bool = False,
        **kwargs,
    ):
        super().__init__()

        if d_model % nhead != 0:
            LOGGER.warning(
                f"d_model ({d_model}) is not divisible by nhead ({nhead}). "
                "This might affect model performance."
            )

        pooling = pooling.lower()
        if pooling not in ["mean", "cls_token", "cls_jet"]:
            raise ValueError(
                f"pooling must be one of 'mean', 'cls_token', or 'cls_jet'. Found: {pooling}."
            )

        if pooling == "cls_jet" and (jet_dim is None or jet_dim <= 0):
            raise ValueError(
                "'cls_jet' pooling requires jet_dim to be specified and > 0."
            )

        # Store configuration
        self.batch_first = batch_first
        self.pooling = pooling
        self.has_jet_input = (jet_dim is not None) and (jet_dim > 0)
        self.uses_cls_token = pooling in ["cls_token", "cls_jet"]

        # Log configuration
        LOGGER.info(f"Particle feature dimension: {part_dim}")
        LOGGER.info(f"Model dimension: {d_model}")
        LOGGER.info(f"Output dimension: {proj_dim}")
        LOGGER.info(f"Pooling strategy: {pooling}")

        if self.pooling == "cls_jet":
            LOGGER.info(f"Using jet features (dim={jet_dim}) as class token")
        elif self.pooling == "cls_token":
            LOGGER.info(
                f"Using class token randomly initialized with dim={d_model}"
            )

        # Initialize embedding layers
        self.particle_embedding = nn.Linear(part_dim, d_model)
        # self.particle_embedding = nn.Sequential(
        #     nn.Linear(part_dim, 2*d_model),
        #     nn.GELU(),
        #     nn.Linear(2*d_model, 2*d_model),
        #     nn.GELU(),
        #     nn.Linear(2*d_model, d_model),
        # )

        # Initialize jet embedding for cls_jet or learnable class token
        if self.pooling == "cls_jet":
            # self.jet_embedding = nn.Sequential(
            #     nn.Linear(jet_dim, 2*d_model),
            #     nn.GELU(),
            #     nn.Linear(2*d_model, 2*d_model),
            #     nn.GELU(),
            #     nn.Linear(2*d_model, d_model),
            # )
            self.jet_embedding = nn.Linear(jet_dim, d_model)
            self.cls_token = None
        elif self.pooling == "cls_token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(
                self.cls_token, std=0.02
            )  # Initialize with small random values
            self.jet_embedding = None
        else:  # mean pooling
            self.cls_token = None
            self.jet_embedding = None

        # Initialize transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first,
            activation=activation,
            **kwargs,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Initialize projection head
        # self.head = nn.Sequential(
        #     nn.Linear(d_model, d_head_hidden),
        #     nn.GELU(),
        #     nn.Linear(d_head_hidden, d_head_hidden),
        #     nn.GELU(),
        #     nn.Linear(d_head_hidden, proj_dim),
        # )
        num_hidden_layers = max(num_head_layers - 2, 1)
        self.head = DINOHead(
            in_dim=d_model,
            proj_dim=proj_dim,
            fc_params=[(d_head_hidden, 0.0) for n in range(num_hidden_layers)],
            l2_norm=head_l2_norm,
            weight_norm=head_weight_norm,
        )
        
        if xavier_init:
            LOGGER.info("Initializing weights using Xavier initialization.")
            # Initialize weights
            """Initialize weights using transformer-appropriate schemes."""
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    # Use different initialization for different layer types
                    if 'head' in name or 'embedding' in name:
                        # For embedding and head layers, use Xavier uniform
                        nn.init.xavier_uniform_(module.weight)
                    else:
                        # For transformer layers, use normal initialization with proper scaling
                        std = 0.02
                        nn.init.normal_(module.weight, mean=0.0, std=std)
                    
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                        
                elif isinstance(module, nn.LayerNorm):
                    # LayerNorm initialization
                    nn.init.constant_(module.bias, 0)
                    nn.init.constant_(module.weight, 1.0)
                    
                elif isinstance(module, nn.MultiheadAttention):
                    # Specific initialization for attention layers
                    if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                        nn.init.xavier_uniform_(module.in_proj_weight)
                    if hasattr(module, 'out_proj'):
                        nn.init.xavier_uniform_(module.out_proj.weight)
                        if module.out_proj.bias is not None:
                            nn.init.constant_(module.out_proj.bias, 0)
            
            # Special initialization for cls_token if using learnable class token
            if self.cls_token is not None:
                nn.init.normal_(self.cls_token, std=0.02)

    def _process_particles(
        self, particles: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Process particle features through normalization and embedding."""
        particles = particles.to(self.device)
        return self.particle_embedding(particles)

    def _get_class_token(
        self, batch_size: int, jets: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Get class token either from jet embedding or learnable parameter."""
        if not self.uses_cls_token:
            return None

        if self.pooling == "cls_jet":
            if jets is None:
                raise ValueError(
                    "jets must be provided when using 'cls_jet' pooling strategy"
                )
            jets = jets.to(self.device)
            return self.jet_embedding(jets).unsqueeze(1)
        else:  # cls_token
            return self.cls_token.expand(batch_size, -1, -1)

    def _safe_mean_pool(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Perform safe mean pooling with numerical stability."""
        mask_expanded = (~mask).unsqueeze(-1).to(embeddings.dtype)
        masked_embeddings = embeddings * mask_expanded
        sum_embeddings = masked_embeddings.sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp_min(EPS)
        return sum_embeddings / count

    def forward(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor,
        jets: torch.Tensor | None = None,
        use_head: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass through the transformer encoder.

        Args:
            particles: Particle features (batch_size, num_particles, part_dim)
            mask: Padding mask (batch_size, num_particles)
                - True for valid particles, False for padding
            jets: Optional jet features (batch_size, jet_dim)
                 Required for 'cls_jet' pooling, optional otherwise
            use_head: Whether to apply the projection head

        Returns:
            proj: Projected output tensor (batch_size, proj_dim) if use_head=True
            pooled: Pooled output tensor (batch_size, d_model) without the projection head
        """
        mask = ~mask  # Convert to mask format expected by PyTorch (True -> masked)
        batch_size = particles.shape[0]

        # Process particles
        particles_embedded = self._process_particles(particles, mask)

        # Get class token if using cls pooling strategy
        cls_token = self._get_class_token(batch_size, jets)

        # Combine embeddings and update mask if using class token
        if cls_token is not None:
            embedded = torch.cat([cls_token, particles_embedded], dim=1)
            full_mask = F.pad(mask, (1, 0), value=False)  # Don't mask cls token
        else:
            embedded = particles_embedded
            full_mask = mask

        # Apply transformer
        if not self.batch_first:
            embedded = embedded.transpose(0, 1)

        output = self.transformer_encoder(embedded, src_key_padding_mask=full_mask)

        if not self.batch_first:
            output = output.transpose(0, 1)

        output = self.final_norm(output)

        # Get final representation based on pooling strategy
        if self.pooling in ["cls_token", "cls_jet"]:
            # Use class token (always first token)
            rep = output[:, 0]
        else:  # mean pooling
            rep = self._safe_mean_pool(output, full_mask)

        if not use_head:
            return rep
        return self.head(rep), rep

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def freeze_backbone(self):
        """Freeze the transformer encoder layers."""
        # Freeze all parameters
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        # Unfreeze the head
        for param in self.head.parameters():
            param.requires_grad = True

    def reset_head(self):
        """Reset the projection head layers."""
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    
    def freeze_embedding(self):
        """Freeze the embedding layers."""
        for param in self.particle_embedding.parameters():
            param.requires_grad = False
        if self.jet_embedding is not None:
            for param in self.jet_embedding.parameters():
                param.requires_grad = False

    def unfreeze_embedding(self):
        """Unfreeze the embedding layers."""
        for param in self.particle_embedding.parameters():
            param.requires_grad = True
        
        if self.jet_embedding is not None:
            for param in self.jet_embedding.parameters():
                param.requires_grad = True

    def load_embedding_weights(self, state_dict: dict):
        """
        Load embedding weights from a state dictionary.
        Only loads weights for embedding layers, ignoring other modules.

        Args:
            state_dict: Dictionary containing the weights to load (typically from a full model).
        """
        # Extract particle embedding weights
        particle_embedding_state = {}
        for key, value in state_dict.items():
            if key.startswith('particle_embedding.'):
                # Remove the 'particle_embedding.' prefix
                new_key = key[len('particle_embedding.'):]
                particle_embedding_state[new_key] = value
        
        if particle_embedding_state:
            self.particle_embedding.load_state_dict(particle_embedding_state)
            print(f"Loaded particle embedding weights: {list(particle_embedding_state.keys())}")
        else:
            print("No particle embedding weights found in state_dict")
        
        # Extract jet embedding weights (if jet_embedding exists)
        if self.jet_embedding is not None:
            jet_embedding_state = {}
            for key, value in state_dict.items():
                if key.startswith('jet_embedding.'):
                    # Remove the 'jet_embedding.' prefix
                    new_key = key[len('jet_embedding.'):]
                    jet_embedding_state[new_key] = value
            
            if jet_embedding_state:
                self.jet_embedding.load_state_dict(jet_embedding_state)
                print(f"Loaded jet embedding weights: {list(jet_embedding_state.keys())}")
            else:
                print("No jet embedding weights found in state_dict")
        
        print("Embedding weights loaded successfully. Other modules in state_dict were ignored.")