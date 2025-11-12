import math
from typing import Literal
import torch
from torch import nn
import torch.nn.functional as F
from .koleo_regularizer import KoLeoLoss
from utils.logger import LOGGER


class DINOLoss(nn.Module):

    def __init__(
        self,
        out_dim: int,
        num_local_views: int = 2,
        num_global_views: int = 2,
        teacher_temp: float = 0.07,
        warmup_teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        warmup_scheduler: Literal["linear", "cosine"] = "linear",
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        skip_same_view: bool = True,
        koleo_loss_weight: float = 0.0,
        enable_sinkhorn_knopp: bool = False,
        sinkhorn_knopp_niters: int = 3,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.num_local_views = num_local_views
        self.num_global_views = num_global_views
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.training = False
        self.skip_same_view = skip_same_view

        # Teacher temperature warmup setup
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.warmup_teacher_temp = warmup_teacher_temp
        self.current_epoch = 0
        
        if warmup_teacher_temp_epochs != -1:
            # Create temperature schedule from warmup_teacher_temp to teacher_temp
            if warmup_scheduler == "linear":
                self.temp_schedule = torch.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                )
            elif warmup_scheduler == "cosine":
                iters = torch.arange(warmup_teacher_temp_epochs)
                self.temp_schedule = warmup_teacher_temp + 0.5 * (teacher_temp - warmup_teacher_temp) * (
                    1 - torch.cos(math.pi * iters / warmup_teacher_temp_epochs)
                )
            LOGGER.info(
                f"Teacher temperature warmup enabled: {warmup_teacher_temp} -> {teacher_temp} "
                f"over {warmup_teacher_temp_epochs} epochs"
            )
        else:
            self.temp_schedule = None

        self.koleo_loss_weight = koleo_loss_weight
        self.enable_koleo_regularizer = koleo_loss_weight > 0
        if self.enable_koleo_regularizer:
            self.koleo_regularizer = KoLeoLoss()
        else:
            self.koleo_regularizer = None

        self.enable_sinkhorn_knopp = enable_sinkhorn_knopp
        self.sinkhorn_knopp_niters = sinkhorn_knopp_niters

    def get_current_teacher_temp(self) -> float:
        """Get the current teacher temperature based on warmup schedule."""
        if self.temp_schedule is None or self.current_epoch >= self.warmup_teacher_temp_epochs:
            return self.teacher_temp
        else:
            return float(self.temp_schedule[self.current_epoch])

    def step_epoch(self):
        """Increment the epoch counter for temperature warmup."""
        self.current_epoch += 1
        if self.temp_schedule is not None and self.current_epoch < self.warmup_teacher_temp_epochs:
            current_temp = self.get_current_teacher_temp()
            LOGGER.info(f"Epoch {self.current_epoch}: Teacher temperature = {current_temp:.4f}")
            
    def resume_epoch(self, epoch: int):
        self.current_epoch = epoch
        if self.temp_schedule is not None and self.current_epoch < self.warmup_teacher_temp_epochs:
            current_temp = self.get_current_teacher_temp()
            LOGGER.info(f"Resumed at epoch {self.current_epoch}: Teacher temperature = {current_temp:.4f}")

    def compute_sum_loss(self, student_out, teacher_out):
        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if self.skip_same_view and v == iq:
                    continue

                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        return total_loss / n_loss_terms if n_loss_terms > 0 else total_loss

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_backbone_output: torch.Tensor = None,
        teacher_backbone_output: torch.Tensor = None,  # Optional, not used in loss computation
        return_components: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        # Temperature scaling
        student_out = student_output / self.student_temp
        num_student_views = self.num_local_views + self.num_global_views
        student_out = student_out.chunk(num_student_views)

        # Teacher processing
        if self.enable_sinkhorn_knopp:
            teacher_out = self.sinkhorn_knopp_teacher(teacher_output)
        else:
            teacher_out = self.softmax_center_teacher(teacher_output)
        teacher_out = teacher_out.detach().chunk(self.num_global_views)
        
        components = {}

        # Compute main DINO loss (always using sum aggregation)
        total_loss = self.compute_sum_loss(student_out, teacher_out)
        components["dino_loss"] = total_loss.detach().cpu().item()

        # Add KoLeo regularization (applied to ALL GLOBAL VIEWS of backbone output)
        if self.enable_koleo_regularizer:
            if student_backbone_output is None:
                raise ValueError(
                    "KoLeo regularization is enabled but student_backbone_output is None."
                )
            
            # Extract all global views from student backbone output
            # In DINOv2, student_backbone_output contains: 
            # [global_view1, global_view2, local_view1, ..., local_viewN]
            samples_per_view = student_backbone_output.shape[0] // (self.num_local_views + self.num_global_views)
            all_global_views = student_backbone_output[:samples_per_view * self.num_global_views]
            global_view_chunks = all_global_views.chunk(self.num_global_views)
            koleo_loss =  sum(
                self.koleo_regularizer(chunk) for chunk in global_view_chunks
            ) / self.num_global_views
            total_loss += self.koleo_loss_weight * koleo_loss
            components["koleo_loss"] = koleo_loss.detach().cpu().item()

        if return_components:
            return total_loss, components
        else:
            return total_loss

    @torch.no_grad()
    def apply_center_update(self, teacher_output: torch.Tensor):
        if not self.training:
            return
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output):
        current_temp = self.get_current_teacher_temp()
        out = F.softmax((teacher_output - self.center) / current_temp, dim=-1)
        self.apply_center_update(teacher_output)
        return out

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output: torch.Tensor):
        """
        Adapted from DINOv2 implementation.
        Note: Sinkhorn-Knopp does NOT use centering - the algorithm itself provides normalization.
        """
        current_temp = self.get_current_teacher_temp()
        # Sinkhorn-Knopp algorithm (no centering applied)
        Q = torch.exp(teacher_output / current_temp).t()  # Q is K-by-B
        B = Q.shape[1]  # number of samples
        K = Q.shape[0]  # number of prototypes

        # Normalize Q
        sum_Q = torch.sum(Q)
        if sum_Q == 0:  # Handle edge case
            return torch.ones_like(Q.t()) / K
        Q /= sum_Q

        for it in range(self.sinkhorn_knopp_niters):
            # Normalize rows: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            sum_of_rows = torch.where(
                sum_of_rows == 0, torch.ones_like(sum_of_rows), sum_of_rows
            )
            Q /= sum_of_rows
            Q /= K

            # Normalize columns: total weight per sample must be 1/B
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            sum_of_cols = torch.where(
                sum_of_cols == 0, torch.ones_like(sum_of_cols), sum_of_cols
            )
            Q /= sum_of_cols
            Q /= B

        Q *= B  # Scale so columns sum to 1
        return Q.t()