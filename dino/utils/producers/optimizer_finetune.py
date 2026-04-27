import torch.nn as nn
from torch import optim
import torch
from typing import Any
from ..ckpt import get_checkpoints_path
from ..logger import LOGGER

OPTIMIZER_MAP = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
}


class SAM(optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021).

    Wraps a base optimizer. Each step:
      1. Compute gradient at current weights
      2. Perturb weights by rho * grad / ||grad|| (ascent step)
      3. Compute gradient at perturbed weights
      4. Restore original weights and step with the perturbed gradient

    Config usage:
        optimizer:
          name: SAM
          params:
            lr: 1e-4
            weight_decay: 0.05
            rho: 0.05
            base_optimizer: AdamW
    """

    def __init__(self, params, base_optimizer_cls=optim.AdamW, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho

    @torch.no_grad()
    def first_step(self):
        """Ascend: perturb weights by rho * grad / ||grad||."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        """Descend: restore weights and step with perturbed gradient."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # restore original weights
        self.base_optimizer.step()

    @torch.no_grad()
    def step(self, closure=None):
        """Standard step (for compatibility). Use first_step/second_step for SAM."""
        # Fallback: just do base optimizer step (no sharpness awareness)
        self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.base_optimizer.state_dict()


def _get_layer_id(name: str, num_layers: int) -> int:
    """Assign a layer index to each backbone parameter for layer-wise LR decay.

    Convention (following BEiT / MAE):
      - layer 0  = embedding (particle_embedding, jet_embedding, cls_token, etc.)
      - layer 1…N = transformer encoder layers
      - layer N+1 = final_norm, pooling_network, and anything else after the encoder

    Returns an integer in [0, num_layers + 1].
    """
    if (
        "particle_embedding" in name
        or "jet_embedding" in name
        or "embedding_norm" in name
        or "jet_norm" in name
        or "cls_token" in name
        or "register_tokens" in name
        or "pos_encoding" in name
        or "rel_pos_bias" in name
        or "scale_conditioning" in name
    ):
        return 0

    if "transformer_encoder.layers." in name:
        # e.g. transformer_encoder.layers.3.self_attn.in_proj_weight
        parts = name.split(".")
        idx = parts.index("layers") + 1
        layer_num = int(parts[idx])
        return layer_num + 1

    # final_norm, pooling_network, anything else
    return num_layers + 1


def get_param_groups(
    backbone: nn.Module,
    head: nn.Module,
    config: dict[str, Any],
    optimizer_params: dict[str, Any] | None = None,
) -> list[dict[str, list[nn.Parameter] | float | str]]:
    """
    Create parameter groups with different learning rates for backbone and head.

    Supports layer-wise LR decay when ``optimizer.layer_decay`` is set in config
    (typical values 0.65–0.9). Each transformer layer i gets:
        lr_i = base_lr * backbone_lr_factor * layer_decay^(num_layers + 1 - layer_id)
    where layer_id 0 = embeddings (lowest LR) and layer_id N+1 = final norm (highest).

    Args:
        backbone: The backbone model (pretrained)
        head: The classification head
        config: The training config
        optimizer_params: Optional parameters to override config values

    Returns:
        List of parameter groups with different learning rates
    """
    optimizer_config = config["training"].get("optimizer", {})
    base_params = optimizer_config.get("params", {})

    # Merge with additional params if provided
    if optimizer_params is not None:
        base_params.update(optimizer_params)

    base_lr = base_params.get("lr", 1e-4)
    backbone_lr_factor = optimizer_config.get("backbone_lr_factor", 1.0)
    layer_decay = optimizer_config.get("layer_decay", 1.0)

    if layer_decay < 1.0:
        # Determine number of transformer layers
        num_layers = len(backbone.transformer_encoder.layers)
        LOGGER.info(
            f"Layer-wise LR decay: {layer_decay}, "
            f"{num_layers} transformer layers + embedding + final"
        )

        # Bucket parameters by layer id
        layer_params: dict[int, list[nn.Parameter]] = {}
        for name, param in backbone.named_parameters():
            lid = _get_layer_id(name, num_layers)
            layer_params.setdefault(lid, []).append(param)

        param_groups = []
        num_groups = num_layers + 2  # 0=embed, 1..N=layers, N+1=final
        for lid in sorted(layer_params.keys()):
            scale = layer_decay ** (num_groups - 1 - lid)
            lr = base_lr * backbone_lr_factor * scale
            param_groups.append(
                {
                    "params": layer_params[lid],
                    "lr": lr,
                    "name": f"backbone_layer_{lid}",
                }
            )
            LOGGER.debug(f"  layer {lid}: lr={lr:.2e} (scale={scale:.4f})")
    else:
        param_groups = [
            {
                "params": list(backbone.parameters()),
                "lr": base_lr * backbone_lr_factor,
                "name": "backbone",
            },
        ]

    param_groups.append(
        {"params": list(head.parameters()), "lr": base_lr, "name": "head"}
    )

    return param_groups


def get_optimizer_finetune(
    config: dict[str, Any],
    model_params: list[nn.Parameter] | list[dict[str, Any]],
    optimizer_params: dict[str, Any] | None = None,
) -> optim.Optimizer:
    """
    Get the optimizer from the training config with enhanced learning rate features.

    Args:
        config: The training config
        model_params: Either a list of parameters or parameter groups with learning rates
        optimizer_params: Optional additional optimizer parameters to override config

    Returns:
        Configured optimizer
    """
    optimizer_config = config["training"].get("optimizer", {})
    LOGGER.info(f"Optimizer config: {optimizer_config}")

    optimizer_type = optimizer_config.get("name", "adam").lower()
    base_optimizer_params = optimizer_config.get("params", {})

    # Merge with additional params if provided
    if optimizer_params is not None:
        base_optimizer_params.update(optimizer_params)

    # Set sensible defaults for different optimizers
    if optimizer_type == "adam":
        base_optimizer_params.setdefault("lr", 1e-4)
        base_optimizer_params.setdefault("weight_decay", 1e-4)
    elif optimizer_type == "adamw":
        base_optimizer_params.setdefault("lr", 1e-4)
        base_optimizer_params.setdefault("weight_decay", 0.01)
    elif optimizer_type == "sgd":
        base_optimizer_params.setdefault("lr", 1e-3)
        base_optimizer_params.setdefault("momentum", 0.9)
        base_optimizer_params.setdefault("weight_decay", 1e-4)

    # Handle SAM optimizer (wraps a base optimizer)
    if optimizer_type == "sam":
        base_opt_name = base_optimizer_params.pop("base_optimizer", "adamw").lower()
        rho = base_optimizer_params.pop("rho", 0.05)
        base_opt_cls = OPTIMIZER_MAP.get(base_opt_name, optim.AdamW)
        optimizer = SAM(model_params, base_optimizer_cls=base_opt_cls, rho=rho, **base_optimizer_params)
        LOGGER.info(f"Created SAM optimizer (rho={rho}, base={base_opt_name})")
    else:
        optimizer_class = OPTIMIZER_MAP.get(optimizer_type)
        if optimizer_class is None:
            LOGGER.warning(f"Unsupported optimizer type: {optimizer_type}. Using Adam.")
            optimizer_class = optim.Adam
        optimizer = optimizer_class(model_params, **base_optimizer_params)
    LOGGER.info(f"Created optimizer: {optimizer}")

    # Load optimizer state if specified in config
    epoch = config["training"].get("load_epoch")
    if epoch is not None:
        ckpt_path = get_checkpoints_path(config=config, epoch_num=epoch)
        checkpoint_dict = torch.load(ckpt_path)
        if "optimizer" in checkpoint_dict:
            state_dict = checkpoint_dict["optimizer"]
        elif "optimizer_state_dict" in checkpoint_dict:
            state_dict = checkpoint_dict["optimizer_state_dict"]
        else:
            raise ValueError("Checkpoint file does not contain the optimizer state")
        optimizer.load_state_dict(state_dict)
        LOGGER.info(f"Loaded optimizer state from {ckpt_path}")

    return optimizer
