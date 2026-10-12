# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stage-II uniform-marginal OT with one detached plan per minibatch."""
import json
from pathlib import Path

import numpy as np
import torch

from verl.utils.bert_emd import (
    MASS_QUANTIZATION_SCALE,
    REPRESENTATION_EPSILON,
    exact_flow,
    layer_cost,
)
from verl.utils.rep_distillation import build_compact_rep_distillation_position_mask

class EMDState:
    def __init__(self, config, device):
        method = json.loads(Path(config.emd_method_path).read_text())
        if method["lambda_emd"] is None:
            raise ValueError("lambda_emd must be explicitly set before launch")
        self.coefficient = float(method["lambda_emd"])
        self.marginal_mode = str(method["marginals"])
        self.mass_scale = int(method["mass_quantization_scale"])
        self.representation_epsilon = float(method["representation_epsilon"])
        self.solver = dict(method["solver"])
        if not self.solver or any(value is None for value in self.solver.values()):
            raise ValueError("every solver setting must be explicitly set before launch")
        assert self.coefficient > 0
        if self.marginal_mode != "uniform_fixed":
            raise ValueError("This ablation requires fixed uniform marginals")
        if method["transport"] != "exact_unregularized_balanced_lp":
            raise ValueError("Stage II requires unregularized balanced linear programming")
        if self.mass_scale != MASS_QUANTIZATION_SCALE:
            raise ValueError(f"mass_quantization_scale must be {MASS_QUANTIZATION_SCALE}")
        if self.representation_epsilon != REPRESENTATION_EPSILON:
            raise ValueError(f"representation_epsilon must be {REPRESENTATION_EPSILON}")
        # Kept outside the language-model state dict so vLLM/HF exports see only LM weights.
        # The complete projector/optimizer/marginal state is saved alongside every actor checkpoint.
        from verl.utils.emd_bridge import FrozenEMDBridge
        bridge_path = Path(config.emd_method_path).resolve().parents[1] / "artifacts/bridge.pt"
        bridge = FrozenEMDBridge(bridge_path, method["bridge_sha256"], device,
                                expected_rank=method["bridge_rank"])
        self.projector = bridge.student
        self.a = np.full(28, 1 / 28, dtype=np.float64)
        self.b = np.full(36, 1 / 36, dtype=np.float64)
        self.updates = 0
        self.method = method

    def save(self, local_path):
        path = Path(local_path) / "emd_state.pt"
        torch.save(dict(projector=self.projector.state_dict(), a=self.a, b=self.b,
                        updates=self.updates, method=self.method), path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)

    def load(self, local_path):
        state = torch.load(Path(local_path) / "emd_state.pt", map_location="cpu", weights_only=False)
        if state["method"] != self.method:
            raise ValueError("EMD method configuration differs from checkpoint")
        self.projector.load_state_dict(state["projector"])
        self.a, self.b, self.updates = state["a"], state["b"], state["updates"]


def update_policy(actor, data):
    from verl.utils.emd_teacher_cache import KEY, release
    keys = data.non_tensor_batch.get(KEY, [])
    try:
        return _update_policy(actor, data)
    finally:
        # Both passes and all four optimizer minibatches have used these rows.
        # On failure the stage stops; a fresh smoke regenerates its teacher data.
        release(keys)


def _update_policy(actor, data):
    if torch.distributed.get_world_size() != 1 or actor.config.use_dynamic_bsz:
        raise ValueError("This validated EMD implementation requires one GPU and static microbatches")
    if actor.config.ppo_epochs != 1 or actor.config.use_kl_loss or actor.config.entropy_coeff:
        raise ValueError("Unexpected deviation from the controlled OPD objective")
    actor.actor_module.train()
    state = actor.emd_state
    device = next(state.projector.parameters()).device
    temperature = data.meta_info["temperature"]
    positions = actor.config.rep_distillation_positions
    last_k = int(actor.config.rep_distillation_last_k)
    if positions != "last_k" or last_k != 2000:
        raise ValueError("This variant requires rep_distillation_positions=last_k and last_k=2000")
    metrics = {}

    def metric(key, value):
        metrics.setdefault(key, []).append(float(value))

    def forward(micro):
        inputs = {**micro.batch, **micro.non_tensor_batch}
        _, _, _, _, hs = actor._forward_micro_batch(
            inputs, temperature=temperature, calculate_entropy=False,
            return_response_hidden_repr=True,
            rep_distillation_positions=positions, rep_distillation_layers="all",
            rep_distillation_last_k=last_k)
        from verl.utils.emd_teacher_cache import KEY, load
        teacher = load(inputs[KEY], device) if KEY in inputs else inputs["teacher_last_hidden_repr"]
        compact_mask = build_compact_rep_distillation_position_mask(
            inputs["response_mask"], positions, last_k=last_k
        )
        cost = layer_cost(
            hs,
            teacher,
            compact_mask,
            state.projector,
            epsilon=state.representation_epsilon,
        )
        return inputs, cost

    for mini in data.split(actor.config.ppo_mini_batch_size):
        microbatches = mini.split(actor.config.ppo_micro_batch_size_per_gpu)
        total = len(mini)
        costs, rngs = [], []
        # First pass: batch-global costs, no retained model graph and no parameter update.
        with torch.no_grad():
            for micro in microbatches:
                rngs.append((torch.get_rng_state(), torch.cuda.get_rng_state()))
                micro.to(device)
                inputs, cost = forward(micro)
                costs.append(cost.detach().cpu() * len(micro) / total)
                micro.to("cpu")
                del inputs, cost
        global_cost = torch.stack(costs).sum(0)
        flow = exact_flow(
            global_cost,
            state.a,
            state.b,
            mass_scale=state.mass_scale,
            solver=state.solver,
        ).to(device)
        actor.actor_optimizer.zero_grad(set_to_none=True)
        observed = torch.zeros_like(global_cost)
        for micro, rng in zip(microbatches, rngs):
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1])
            micro.to(device)
            inputs, cost = forward(micro)
            emd = (flow * cost).sum()
            loss = state.coefficient * emd
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Stage-II representation loss")
            scale = len(micro) / total
            (loss * scale).backward()
            observed += cost.detach().cpu() * scale
            metric("actor/emd_loss", emd.detach().item() * scale)
            metric("emd/representation_only", 1.0 * scale)
            metric("emd/normalized_cost", 1.0 * scale)
            metric("emd/last_k_2000", 1.0 * scale)
            micro.to("cpu")
            del inputs, cost, emd, loss
        if not torch.allclose(observed, global_cost, rtol=2e-4, atol=2e-5):
            raise RuntimeError("EMD two-pass cost mismatch; parameters/RNG or token alignment changed")
        # The bridge is frozen; only student gradients are clipped and updated.
        norm = actor.actor_module.clip_grad_norm_(float("inf"))
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite student gradient")
        coefficient = (float(actor.config.grad_clip) / (norm + 1e-6)).clamp(max=1.)
        for param in actor.actor_module.parameters():
            if param.grad is not None: param.grad.mul_(coefficient)
        actor.actor_optimizer.step()
        state.updates += 1
        student_support = 1.0 / np.square(state.a).sum()
        teacher_support = 1.0 / np.square(state.b).sum()
        metric("actor/grad_norm", norm.item())
        metric("emd/optimizer_updates", state.updates)
        metric("emd/uniform_marginals", 1.0)
        metric("emd/student_effective_support", student_support)
        metric("emd/teacher_effective_support", teacher_support)
        metric("emd/student_max_mass", state.a.max())
        metric("emd/teacher_max_mass", state.b.max())
    actor.actor_optimizer.zero_grad(set_to_none=True)
    return metrics
