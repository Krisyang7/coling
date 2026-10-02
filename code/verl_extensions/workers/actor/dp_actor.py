# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from contextlib import nullcontext

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.dlt_opd import (
    DLTConfig,
    FrozenSharedBridge,
    extract_dlt_response_repr,
    layer_transport_loss,
    load_frozen_plan,
    select_uniform_response_tokens,
)
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.att_distillation import (
    att_distillation_query_valid_fraction,
    attention_distillation_loss,
    disable_gradient_checkpointing,
    forward_response_attn_rows,
    get_att_distillation_context_metadata,
    validate_att_distillation_loss_type,
)
from verl.utils.rep_distillation import (
    LowRankCrossArchProjector,
    DirectLowRankCrossArchProjector,
    ResidualLowRankCrossArchProjector,
    infer_subspace_mode_from_checkpoint,
    load_preexp_checkpoint_dict,
    resolve_student_projector_settings,
    _squeeze_hidden_rmpad,
    build_compact_rep_distillation_position_mask,
    build_rep_distillation_position_mask,
    compact_response_repr_by_positions,
    extract_teacher_response_hidden_repr,
    forward_response_hidden_repr,
    hidden_states_tuple_to_response_repr,
    compute_rep_alignment_metrics,
    multi_layer_normalized_cosine_similarity,
    multi_layer_normalized_mse_loss,
    validate_rep_distillation_layers,
    validate_rep_distillation_positions,
    validate_rep_projector_mode,
)
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.rep_projector = None
        self._rep_projector_in_optimizer = False
        self.low_rank_projector: LowRankCrossArchProjector | DirectLowRankCrossArchProjector | None = None
        self._low_rank_projector_in_optimizer = False
        self.residual_low_rank_projector: ResidualLowRankCrossArchProjector | None = None
        self._residual_low_rank_projector_in_optimizer = False
        from verl.workers.actor.emd_update import EMDState
        self.emd_state = EMDState(self.config, get_device_id())
        self.dlt_bridge: FrozenSharedBridge | None = None
        self.dlt_frozen_plan: torch.Tensor | None = None

    def _get_dlt_bridge(self) -> FrozenSharedBridge:
        if self.dlt_bridge is None:
            checkpoint = self.config.get("dlt_bridge_checkpoint", None)
            if not checkpoint:
                raise ValueError("DLT requires dlt_bridge_checkpoint")
            self.dlt_bridge = FrozenSharedBridge.load(checkpoint).to(get_device_id())
            self.dlt_bridge.eval()
            if self.config.get("dlt_mapping_mode", "live_ot") == "frozen_ot":
                plan_checkpoint = self.config.get("dlt_frozen_plan_checkpoint", None)
                if not plan_checkpoint:
                    raise ValueError("frozen_ot requires dlt_frozen_plan_checkpoint")
                self.dlt_frozen_plan = load_frozen_plan(plan_checkpoint)
        return self.dlt_bridge

    def _get_or_create_rep_projector(self, student_dim: int, teacher_dim: int) -> nn.Linear | None:
        if student_dim == teacher_dim:
            return None
        if self.rep_projector is None:
            self.rep_projector = nn.Linear(student_dim, teacher_dim, bias=False).to(get_device_id())
            if self.actor_optimizer is not None and not self._rep_projector_in_optimizer:
                self.actor_optimizer.add_param_group({"params": self.rep_projector.parameters()})
                self._rep_projector_in_optimizer = True
        return self.rep_projector

    def _get_or_create_low_rank_projector(
        self,
        student_dim: int,
        teacher_dim: int,
        *,
        num_layers: int,
        num_teacher_layers: int,
    ) -> LowRankCrossArchProjector | DirectLowRankCrossArchProjector:
        if self.low_rank_projector is None:
            rank = int(self.config.get("rep_low_rank", 256))
            checkpoint = self.config.get("rep_low_rank_init_checkpoint", None)
            ps_type, mlp_mult, mlp_hidden_dim = resolve_student_projector_settings(
                ps_projector=str(self.config.get("rep_ps_projector", "auto")),
                mlp_hidden_mult=int(self.config.get("rep_mlp_hidden_mult", 4)),
                checkpoint_path=checkpoint,
            )
            subspace_mode = "full"
            if checkpoint:
                subspace_mode = infer_subspace_mode_from_checkpoint(
                    load_preexp_checkpoint_dict(checkpoint)
                )
            if subspace_mode == "direct":
                if ps_type != "linear":
                    raise ValueError(
                        f"Direct preexp2 checkpoint at {checkpoint} requires linear P_S, got {ps_type!r}"
                    )
                self.low_rank_projector = DirectLowRankCrossArchProjector(
                    num_layers=num_layers,
                    student_dim=student_dim,
                    teacher_dim=teacher_dim,
                    rank=rank,
                    num_teacher_layers=num_teacher_layers,
                ).to(get_device_id())
            else:
                self.low_rank_projector = LowRankCrossArchProjector(
                    num_layers=num_layers,
                    student_dim=student_dim,
                    teacher_dim=teacher_dim,
                    rank=rank,
                    num_teacher_layers=num_teacher_layers,
                    ps_projector_type=ps_type,
                    mlp_hidden_mult=mlp_mult,
                    mlp_hidden_dim=mlp_hidden_dim,
                ).to(get_device_id())
            if checkpoint:
                self.low_rank_projector.load_from_preexp_checkpoint(checkpoint)
            freeze_ps = bool(self.config.get("rep_freeze_ps", False))
            if freeze_ps:
                self.low_rank_projector.freeze_student_projectors()
                if self.low_rank_projector._loaded_ps_layers == 0:
                    print(
                        "WARNING: rep_freeze_ps=True but no P_S weights were loaded from checkpoint. "
                        "Rep loss will use random frozen P_S; set rep_low_rank_init_checkpoint."
                    )
            elif self.actor_optimizer is not None and not self._low_rank_projector_in_optimizer:
                trainable_params = self.low_rank_projector.trainable_parameters()
                if trainable_params:
                    self.actor_optimizer.add_param_group({"params": trainable_params})
                    self._low_rank_projector_in_optimizer = True
        return self.low_rank_projector

    def _get_or_create_residual_low_rank_projector(
        self,
        student_dim: int,
        teacher_dim: int,
        *,
        num_layers: int,
        num_teacher_layers: int,
    ) -> ResidualLowRankCrossArchProjector:
        if self.residual_low_rank_projector is None:
            head_rank = int(self.config.get("rep_head_rank", 16))
            tail_rank = int(self.config.get("rep_low_rank", 16))
            head_checkpoint = self.config.get("rep_head_init_checkpoint", None)
            tail_checkpoint = self.config.get("rep_low_rank_init_checkpoint", None)
            if not head_checkpoint:
                raise ValueError(
                    "rep_projector_mode=low_rank_residual requires rep_head_init_checkpoint "
                    "(pre-experiment 2 head ps_bank.pt, e.g. rank_16/ps_bank.pt)"
                )
            tail_ps_type, tail_mlp_mult, tail_mlp_hidden_dim = resolve_student_projector_settings(
                ps_projector=str(self.config.get("rep_ps_projector", "auto")),
                mlp_hidden_mult=int(self.config.get("rep_mlp_hidden_mult", 4)),
                checkpoint_path=tail_checkpoint,
            )
            self.residual_low_rank_projector = ResidualLowRankCrossArchProjector(
                num_layers=num_layers,
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                head_rank=head_rank,
                tail_rank=tail_rank,
                num_teacher_layers=num_teacher_layers,
                tail_ps_projector_type=tail_ps_type,
                tail_mlp_hidden_mult=tail_mlp_mult,
                tail_mlp_hidden_dim=tail_mlp_hidden_dim,
            ).to(get_device_id())
            self.residual_low_rank_projector.load_head_from_preexp_checkpoint(head_checkpoint)
            if tail_checkpoint:
                self.residual_low_rank_projector.load_tail_from_preexp_checkpoint(tail_checkpoint)
            freeze_ps = bool(self.config.get("rep_freeze_ps", False))
            if freeze_ps:
                self.residual_low_rank_projector.freeze_student_projectors()
                if self.residual_low_rank_projector._loaded_tail_ps_layers == 0:
                    print(
                        "WARNING: rep_freeze_ps=True but no tail P_S loaded. "
                        "Set rep_low_rank_init_checkpoint to a residual ps_tail_bank.pt."
                    )
            elif self.actor_optimizer is not None and not self._residual_low_rank_projector_in_optimizer:
                trainable_params = self.residual_low_rank_projector.trainable_parameters()
                if trainable_params:
                    self.actor_optimizer.add_param_group({"params": trainable_params})
                    self._residual_low_rank_projector_in_optimizer = True
        return self.residual_low_rank_projector

    def _extract_response_hidden_repr_from_output(
        self,
        output,
        response_mask: torch.Tensor,
        *,
        indices=None,
        batch_size: int | None = None,
        seqlen: int | None = None,
        pad_size: int = 0,
        rep_distillation_positions: str = "all",
        rep_distillation_layers: str = "last",
        rep_distillation_last_k: int = 32,
        rep_distillation_first_k: int = 50,
    ) -> torch.Tensor:
        if self.config.get("use_dlt", False):
            repr, _ = extract_dlt_response_repr(
                output.hidden_states,
                response_mask,
                layer_count=int(self.config.get("dlt_student_layers", 8)),
                token_count=int(self.config.get("dlt_token_count", 512)),
                indices=indices,
                batch_size=batch_size,
                seqlen=seqlen,
                use_ulysses_sp=self.use_ulysses_sp,
                pad_size=pad_size,
                use_remove_padding=self.use_remove_padding,
            )
            return repr
        if rep_distillation_layers != "last":
            if self.use_remove_padding:
                return hidden_states_tuple_to_response_repr(
                    output.hidden_states,
                    response_mask,
                    rep_distillation_positions,
                    rep_distillation_layers,
                    indices=indices,
                    batch_size=batch_size,
                    seqlen=seqlen,
                    use_ulysses_sp=self.use_ulysses_sp,
                    pad_size=pad_size,
                    use_remove_padding=True,
                    last_k=rep_distillation_last_k,
                    first_k=rep_distillation_first_k,
                )
            return extract_teacher_response_hidden_repr(
                output.hidden_states,
                response_mask,
                rep_distillation_positions,
                rep_distillation_layers,
                last_k=rep_distillation_last_k,
                first_k=rep_distillation_first_k,
            )

        if self.use_remove_padding:
            last_hidden_rmpad = _squeeze_hidden_rmpad(output.hidden_states[-1])
            if self.use_ulysses_sp:
                last_hidden_rmpad = gather_outputs_and_unpad(
                    last_hidden_rmpad,
                    gather_dim=0,
                    unpad_dim=0,
                    padding_size=pad_size,
                )
            full_last_hidden = pad_input(
                hidden_states=last_hidden_rmpad,
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            )
            hidden_states = full_last_hidden.float()
        else:
            hidden_states = output.hidden_states[-1].float()

        return extract_teacher_response_hidden_repr(
            hidden_states,
            response_mask,
            rep_distillation_positions,
            "last",
            last_k=rep_distillation_last_k,
            first_k=rep_distillation_first_k,
        )

    def _forward_response_hidden_repr(self, micro_batch) -> torch.Tensor:
        if self.config.get("use_dlt", False):
            _, _, _, _, response_repr = self._forward_micro_batch(
                micro_batch,
                temperature=1.0,
                return_response_hidden_repr=True,
            )
            return response_repr
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        rep_distillation_positions = validate_rep_distillation_positions(
            self.config.get("rep_distillation_positions", "last")
        )
        rep_distillation_layers = validate_rep_distillation_layers(
            self.config.get("rep_distillation_layers", "last")
        )
        rep_distillation_last_k = int(self.config.get("rep_distillation_last_k", 32))
        rep_distillation_first_k = int(self.config.get("rep_distillation_first_k", 50))

        return forward_response_hidden_repr(
            self.actor_module,
            micro_batch["input_ids"],
            micro_batch["attention_mask"],
            micro_batch["position_ids"],
            micro_batch["response_mask"],
            use_remove_padding=self.use_remove_padding,
            use_ulysses_sp=self.use_ulysses_sp,
            ulysses_sequence_parallel_size=self.ulysses_sequence_parallel_size,
            multi_modal_inputs=multi_modal_inputs,
            positions=rep_distillation_positions,
            layers=rep_distillation_layers,
            last_k=rep_distillation_last_k,
            first_k=rep_distillation_first_k,
        )

    def _forward_response_attn_rows(self, micro_batch) -> torch.Tensor:
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        att_distillation_positions = validate_rep_distillation_positions(
            self.config.get("att_distillation_positions", "last")
        )
        att_distillation_layers = validate_rep_distillation_layers(
            self.config.get("att_distillation_layers", "last")
        )

        return forward_response_attn_rows(
            self.actor_module,
            micro_batch["input_ids"],
            micro_batch["attention_mask"],
            micro_batch["position_ids"],
            micro_batch["response_mask"],
            multi_modal_inputs=multi_modal_inputs,
            positions=att_distillation_positions,
            layers=att_distillation_layers,
            last_k=int(self.config.get("att_distillation_last_k", 32)),
            first_k=int(self.config.get("att_distillation_first_k", 50)),
            max_context_len=int(self.config.get("att_distillation_max_key_len", 4096)),
        )

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        top_k=0,
        student_top_k_ids=None,
        return_response_hidden_repr: bool = False,
        rep_distillation_positions: str = "all",
        rep_distillation_layers: str = "last",
        rep_distillation_last_k: int = 32,
        rep_distillation_first_k: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
            response_hidden_repr: # (bs, response_len, hidden_dim) or None
        """
        response_length = micro_batch["responses"].size(-1)
        response_mask = None
        if return_response_hidden_repr:
            if "response_mask" in micro_batch:
                response_mask = micro_batch["response_mask"]
            else:
                response_mask = micro_batch["attention_mask"][:, -response_length:]
        response_hidden_repr = None
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                pad_size = 0
                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=return_response_hidden_repr,
                    **extra_args,
                )  # prevent model thinks we are generating

                if return_response_hidden_repr:
                    response_hidden_repr = self._extract_response_hidden_repr_from_output(
                        output,
                        response_mask,
                        indices=indices,
                        batch_size=batch_size,
                        seqlen=seqlen,
                        pad_size=pad_size,
                        rep_distillation_positions=rep_distillation_positions,
                        rep_distillation_layers=rep_distillation_layers,
                        rep_distillation_last_k=rep_distillation_last_k,
                        rep_distillation_first_k=rep_distillation_first_k,
                    )

                need_logits = top_k > 0

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False

                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation and gradient
                    # issues from inplace operations
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        # Note: we don't use inplace_backward here to ensure correct gradients
                        # when both log_probs and topk_log_probs are needed
                        log_probs_all = torch.log_softmax(logits_rmpad, dim=-1)
                        # Gather log_probs for target tokens
                        log_probs = log_probs_all.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                    if need_topk:
                        if student_top_k_ids is not None:
                             # Use specific IDs (from rollout)
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3: # (bsz, seqlen, k)
                                 # We are in rmpad mode, but student_top_k_ids is padded 3D tensor
                                 # We need to extract the relevant tokens aligning with input_ids_rmpad_rolled

                                 # This is tricky because student_top_k_ids is shaped (batch, seq, k)
                                 # and logits_rmpad is (total_nnz, vocab)
                                 # We need to flatten student_top_k_ids to (total_nnz, k) using indices

                                 # Re-use the indices computed from unpad_input
                                 # indices: (total_nnz,)
                                 # student_top_k_ids: (batch, seq, k)

                                 # 1. If student_top_k_ids only covers the response, pad it to match full sequence length
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k),
                                                                         dtype=student_top_k_ids.dtype,
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 # 2. Flatten student_top_k_ids to (batch*seq, k)
                                 flat_ids = student_top_k_ids.view(-1, top_k)

                                 # 3. Select using indices
                                 # Note: indices are from attention_mask, which aligns with how logits_rmpad represents data
                                 topk_ids_rmpad = flat_ids[indices] # (total_nnz, k)

                                 # If 'student_top_k_ids' in batch has shape (batch, seq_len, k), then:
                                 topk_ids = topk_ids_rmpad

                             else:
                                 # If it's already flattened? Unlikely.
                                 pass

                        else:
                             # Legacy/Resample behavior
                             _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=return_response_hidden_repr,
                    **extra_args,
                )  # prevent model thinks we are generating

                if return_response_hidden_repr:
                    response_hidden_repr = self._extract_response_hidden_repr_from_output(
                        output,
                        response_mask,
                        rep_distillation_positions=rep_distillation_positions,
                        rep_distillation_layers=rep_distillation_layers,
                        rep_distillation_last_k=rep_distillation_last_k,
                        rep_distillation_first_k=rep_distillation_first_k,
                    )

                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        log_probs_all = torch.log_softmax(logits, dim=-1)
                        # Gather log_probs for target tokens (responses)
                        log_probs = log_probs_all.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)

                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

            return entropy, log_probs, topk_ids, topk_log_probs, response_hidden_repr

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses,
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs, _ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False,
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("log_prob_top_k", 0)
        strategy = data.meta_info.get("top_k_strategy", "only_stu")
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")  # "student_p", "teacher_p", or "none"
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]

            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching,
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"],
                                 non_tensor_batch_keys=non_tensor_select_keys)

            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs, _ = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False,
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)

        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)

        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            """Compute weights for reward calculation.

            Args:
                S_logp: Student log probabilities (batch, seq, K)
                T_logp: Teacher log probabilities (batch, seq, K)
                valid_mask: Boolean mask for valid tokens (batch, seq, K)
                weight_mode: "student_p", "teacher_p", or "none"
                normalize: If True, apply softmax normalization across K dim.
                          If False, use raw probabilities (masked by valid_mask).

            Returns:
                Weights (batch, seq, K)
            """
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode == "teacher_p":
                log_probs = T_logp
            elif weight_mode == "none":
                # 对于"none"模式，使用均匀分布
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")

            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))

            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)

            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

            return weights

        res_tensors = {}

        if strategy == "only_stu":
            kl_val = S_logp - T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights

        elif strategy == "only_tch":
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors["union_top_k_ids"] = T_ids

        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights

        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)

            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights

            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T

        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,    # S_ids is valid if not in T
                ~T_in_S     # T_ids is valid if not in S
            ], dim=-1)

            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = -kl_val * norm_weights

            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T

        res_tensors["rm_scores"] = rm_scores
        return DataProto.from_dict(tensors=res_tensors)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            self.actor_optimizer.zero_grad()
            raise FloatingPointError(f"Non-finite actor gradient: {grad_norm}")
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, topk_ids, topk_log_probs, _ = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, top_k=top_k
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        from verl.workers.actor.emd_update import update_policy
        return update_policy(self, data)
