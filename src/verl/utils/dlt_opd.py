"""Dynamic layer transport utilities for on-policy distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

MappingMode = Literal["fixed", "uniform", "frozen_ot", "live_ot"]


@dataclass(frozen=True)
class DLTConfig:
    mapping_mode: MappingMode = "live_ot"
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iters: int = 30
    sinkhorn_max_iters: int = 50
    marginal_tolerance: float = 1e-3
    cost_scale_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.mapping_mode not in {"fixed", "uniform", "frozen_ot", "live_ot"}:
            raise ValueError(f"unsupported mapping_mode: {self.mapping_mode}")
        if self.sinkhorn_epsilon <= 0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if self.sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be positive")
        if self.sinkhorn_max_iters < self.sinkhorn_iters:
            raise ValueError("sinkhorn_max_iters must be >= sinkhorn_iters")
        if self.marginal_tolerance <= 0:
            raise ValueError("marginal_tolerance must be positive")


class FrozenSharedBridge(nn.Module):
    """One frozen student/teacher projection shared by all selected layers."""

    FORMAT = "dlt-opd-shared-bridge-v1"

    def __init__(
        self,
        student_weight: torch.Tensor,
        teacher_weight: torch.Tensor,
        teacher_mean: torch.Tensor,
    ) -> None:
        super().__init__()
        if student_weight.ndim != 2 or teacher_weight.ndim != 2:
            raise ValueError("bridge weights must have shape (rank, hidden_dim)")
        if student_weight.size(0) != teacher_weight.size(0):
            raise ValueError("student and teacher bridge ranks differ")
        if teacher_mean.shape != (teacher_weight.size(1),):
            raise ValueError("teacher_mean shape does not match teacher hidden dimension")
        self.register_buffer("student_weight", student_weight.detach().float().clone())
        self.register_buffer("teacher_weight", teacher_weight.detach().float().clone())
        self.register_buffer("teacher_mean", teacher_mean.detach().float().clone())

    @property
    def rank(self) -> int:
        return int(self.student_weight.size(0))

    def project_student(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.student_weight.to(device=hidden.device, dtype=hidden.dtype))

    def project_teacher(self, hidden: torch.Tensor) -> torch.Tensor:
        mean = self.teacher_mean.to(device=hidden.device, dtype=hidden.dtype)
        weight = self.teacher_weight.to(device=hidden.device, dtype=hidden.dtype)
        return F.linear(hidden - mean, weight)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "FrozenSharedBridge":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if checkpoint.get("format") != cls.FORMAT:
            raise ValueError(f"unsupported bridge checkpoint format: {checkpoint.get('format')!r}")
        return cls(
            checkpoint["student_weight"],
            checkpoint["teacher_weight"],
            checkpoint["teacher_mean"],
        )


def select_spaced_hidden_state_indices(num_hidden_states: int, count: int) -> list[int]:
    """Select block outputs, excluding embeddings and including the final block."""
    num_layers = num_hidden_states - 1
    if num_layers <= 0 or count <= 0 or count > num_layers:
        raise ValueError(f"cannot select {count} transformer layers from {num_layers}")
    if count == num_layers:
        return list(range(1, num_hidden_states))
    indices = torch.linspace(1, num_layers, steps=count).round().long().tolist()
    indices[-1] = num_layers
    if len(set(indices)) != count:
        raise ValueError(f"cannot select {count} distinct layers from {num_layers}")
    return indices


def select_uniform_response_tokens(
    response_mask: torch.Tensor,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select uniformly spaced valid response positions with fixed-width padding."""
    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape (B, T)")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    batch_size = response_mask.size(0)
    indices = torch.zeros(batch_size, max_tokens, dtype=torch.long, device=response_mask.device)
    selected_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=response_mask.device)
    for batch_idx in range(batch_size):
        valid = torch.nonzero(response_mask[batch_idx].bool(), as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        take = min(max_tokens, int(valid.numel()))
        offsets = torch.linspace(0, valid.numel() - 1, steps=take, device=valid.device).round().long()
        indices[batch_idx, :take] = valid.index_select(0, offsets)
        selected_mask[batch_idx, :take] = True
    return indices, selected_mask


def extract_dlt_response_repr(
    hidden_states: tuple[torch.Tensor, ...],
    response_mask: torch.Tensor,
    *,
    layer_count: int,
    token_count: int,
    indices: torch.Tensor | None = None,
    batch_size: int | None = None,
    seqlen: int | None = None,
    use_ulysses_sp: bool = False,
    pad_size: int = 0,
    ulysses_group=None,
    use_remove_padding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(B, L, K, D)`` DLT states and their ``(B, K)`` mask."""
    layer_indices = select_spaced_hidden_state_indices(len(hidden_states), layer_count)
    if use_remove_padding:
        from verl.utils.attention_utils import pad_input
        from verl.utils.rep_distillation import _squeeze_hidden_rmpad
        from verl.utils.ulysses import gather_outputs_and_unpad

    token_indices, selected_mask = select_uniform_response_tokens(response_mask, token_count)
    response_len = response_mask.size(1)
    batch_indices = torch.arange(response_mask.size(0), device=response_mask.device).unsqueeze(1)
    layer_reprs = []
    for layer_idx in layer_indices:
        layer_hidden = hidden_states[layer_idx]
        if use_remove_padding:
            if indices is None or batch_size is None or seqlen is None:
                raise ValueError("rmpad metadata is required when use_remove_padding=True")
            hidden_rmpad = _squeeze_hidden_rmpad(layer_hidden)
            if use_ulysses_sp:
                hidden_rmpad = gather_outputs_and_unpad(
                    hidden_rmpad,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                    group=ulysses_group,
                )
            layer_hidden = pad_input(
                hidden_states=hidden_rmpad,
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            )
        response_hidden = layer_hidden.float()[:, -response_len:, :]
        selected = response_hidden[batch_indices, token_indices]
        selected = selected * selected_mask.unsqueeze(-1).to(selected.dtype)
        layer_reprs.append(selected)
    return torch.stack(layer_reprs, dim=1), selected_mask


def pairwise_layer_cost(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    if student_repr.ndim != 4 or teacher_repr.ndim != 4:
        raise ValueError("student_repr and teacher_repr must have shape (B, L, K, D)")
    if student_repr.shape[0] != teacher_repr.shape[0] or student_repr.shape[2:] != teacher_repr.shape[2:]:
        raise ValueError("student and teacher batch/token/rank dimensions must match")
    if position_mask.shape != student_repr.shape[:1] + student_repr.shape[2:3]:
        raise ValueError("position_mask must have shape (B, K)")
    weights = position_mask.to(student_repr.dtype)
    valid_counts = weights.sum(dim=-1)
    if torch.any(valid_counts <= 0):
        raise ValueError("every trajectory must contain at least one selected response token")
    student_norm = F.normalize(student_repr, p=2, dim=-1)
    teacher_norm = F.normalize(teacher_repr.detach(), p=2, dim=-1)
    cosine = torch.einsum("bskd,btkd->bstk", student_norm, teacher_norm)
    return ((1.0 - cosine) * weights[:, None, None, :]).sum(-1) / valid_counts[:, None, None]


def proportional_fixed_plan(cost: torch.Tensor) -> torch.Tensor:
    batch_size, num_student_layers, num_teacher_layers = cost.shape
    if num_student_layers == 1:
        teacher_indices = torch.tensor([num_teacher_layers - 1], device=cost.device)
    else:
        teacher_indices = torch.linspace(
            0, num_teacher_layers - 1, steps=num_student_layers, device=cost.device
        ).round().long()
    plan = torch.zeros(num_student_layers, num_teacher_layers, device=cost.device, dtype=cost.dtype)
    plan[torch.arange(num_student_layers, device=cost.device), teacher_indices] = 1.0 / num_student_layers
    return plan.unsqueeze(0).expand(batch_size, -1, -1)


def uniform_plan(cost: torch.Tensor) -> torch.Tensor:
    _, num_student_layers, num_teacher_layers = cost.shape
    return torch.full_like(cost, 1.0 / (num_student_layers * num_teacher_layers))


def marginal_residual(plan: torch.Tensor) -> torch.Tensor:
    num_student_layers, num_teacher_layers = plan.shape[-2:]
    row_error = (plan.sum(-1) - 1.0 / num_student_layers).abs().amax(-1)
    col_error = (plan.sum(-2) - 1.0 / num_teacher_layers).abs().amax(-1)
    return torch.maximum(row_error, col_error)


def log_sinkhorn_plan(cost: torch.Tensor, config: DLTConfig) -> tuple[torch.Tensor, torch.Tensor, int]:
    if cost.ndim != 3 or not torch.isfinite(cost).all():
        raise ValueError("cost must be a finite (B, Ls, Lt) tensor")
    _, num_student_layers, num_teacher_layers = cost.shape
    scale = cost.detach().flatten(1).median(1).values.clamp_min(config.cost_scale_epsilon)
    log_kernel = -(cost.detach() / scale[:, None, None]) / config.sinkhorn_epsilon
    log_a = cost.new_full((cost.size(0), num_student_layers), -torch.log(cost.new_tensor(float(num_student_layers))))
    log_b = cost.new_full((cost.size(0), num_teacher_layers), -torch.log(cost.new_tensor(float(num_teacher_layers))))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    iterations = config.sinkhorn_max_iters
    for step in range(config.sinkhorn_max_iters):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[:, None, :], dim=-1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, :, None], dim=-2)
        if step + 1 == config.sinkhorn_iters:
            plan = torch.exp(log_kernel + log_u[:, :, None] + log_v[:, None, :])
            if torch.all(marginal_residual(plan) <= config.marginal_tolerance):
                iterations = step + 1
                break
    plan = torch.exp(log_kernel + log_u[:, :, None] + log_v[:, None, :])
    return plan.detach(), marginal_residual(plan).detach(), iterations


def load_frozen_plan(path: str | Path) -> torch.Tensor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "dlt-opd-frozen-plan-v1":
        raise ValueError(f"unsupported frozen plan checkpoint: {checkpoint.get('format')!r}")
    return checkpoint["plan"].detach().float()


def layer_transport_loss(
    student_repr: torch.Tensor,
    teacher_repr: torch.Tensor,
    position_mask: torch.Tensor,
    *,
    config: DLTConfig,
    frozen_plan: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    cost = pairwise_layer_cost(student_repr, teacher_repr, position_mask)
    if not torch.isfinite(cost).all():
        raise ValueError("layer cost contains NaN or Inf")
    if config.mapping_mode == "fixed":
        plan = proportional_fixed_plan(cost)
        residual = marginal_residual(plan).detach()
        iterations = 0
    elif config.mapping_mode == "uniform":
        plan = uniform_plan(cost)
        residual = marginal_residual(plan).detach()
        iterations = 0
    elif config.mapping_mode == "frozen_ot":
        if frozen_plan is None:
            raise ValueError("frozen_ot requires dlt_frozen_plan_checkpoint")
        plan = frozen_plan.to(device=cost.device, dtype=cost.dtype)
        if plan.ndim == 2:
            plan = plan.unsqueeze(0)
        if plan.shape[-2:] != cost.shape[-2:] or plan.size(0) not in (1, cost.size(0)):
            raise ValueError("frozen plan dimensions do not match the current layer cost")
        if not torch.isfinite(plan).all() or torch.any(plan < 0):
            raise ValueError("frozen plan must be finite and non-negative")
        if torch.any(plan.sum(dim=(-2, -1)) <= 0):
            raise ValueError("frozen plan must have positive mass")
        plan = (plan / plan.sum(dim=(-2, -1), keepdim=True)).expand(cost.size(0), -1, -1).detach()
        residual = marginal_residual(plan).detach()
        if torch.any(residual > config.marginal_tolerance):
            raise ValueError("frozen plan does not satisfy uniform layer marginals")
        iterations = 0
    else:
        plan, residual, iterations = log_sinkhorn_plan(cost, config)

    loss = (plan.detach() * cost).sum(dim=(-2, -1)).mean()
    entropy = -(plan.clamp_min(1e-12) * plan.clamp_min(1e-12).log()).sum(dim=(-2, -1)).mean()
    metrics = {
        "dlt/loss": float(loss.detach()),
        "dlt/cost_mean": float(cost.detach().mean()),
        "dlt/cost_contrast": float((cost.detach().amax((-2, -1)) - cost.detach().amin((-2, -1))).mean()),
        "dlt/hidden_alignment": float(1.0 - cost.detach().mean()),
        "dlt/plan_entropy": float(entropy.detach()),
        "dlt/marginal_residual": float(residual.max()),
        "dlt/sinkhorn_iterations": float(iterations),
    }
    mean_plan = plan.detach().mean(0).float().cpu()
    mean_cost = cost.detach().mean(0).float().cpu()
    student_norm = student_repr.detach().norm(dim=-1).mean(dim=(0, 2)).float().cpu()
    teacher_norm = teacher_repr.detach().norm(dim=-1).mean(dim=(0, 2)).float().cpu()
    for student_idx in range(mean_plan.size(0)):
        metrics[f"dlt/student_layer_{student_idx}_best_cost"] = float(mean_cost[student_idx].min())
        metrics[f"dlt/student_layer_{student_idx}_norm"] = float(student_norm[student_idx])
        for teacher_idx in range(mean_plan.size(1)):
            metrics[f"dlt/plan_s{student_idx}_t{teacher_idx}"] = float(mean_plan[student_idx, teacher_idx])
    for teacher_idx in range(mean_plan.size(1)):
        metrics[f"dlt/teacher_layer_{teacher_idx}_best_cost"] = float(mean_cost[:, teacher_idx].min())
        metrics[f"dlt/teacher_layer_{teacher_idx}_norm"] = float(teacher_norm[teacher_idx])
    return loss, cost.detach(), plan.detach(), metrics
