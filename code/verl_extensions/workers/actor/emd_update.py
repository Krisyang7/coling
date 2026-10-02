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
"""Single-GPU Bridge + EMD + OPD, with one fixed plan per optimizer minibatch."""
import json
from pathlib import Path

import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.utils.bert_emd import exact_flow, layer_cost, update_marginals


def combine_objective(pg, emd, coefficient, representation_only):
    """Select one unambiguous optimization objective for the current stage."""
    if representation_only:
        return coefficient * emd
    return pg + coefficient * emd


class EMDState:
    def __init__(self, config, device):
        method = json.loads(Path(config.emd_method_path).read_text())
        if method["lambda_emd"] is None or method["tau"] is None:
            raise ValueError("EMD-specific parameters must be explicitly set before launch")
        self.coefficient = float(method["lambda_emd"])
        self.tau = float(method["tau"])
        assert self.coefficient > 0 and self.tau > 0
        # Kept outside the language-model state dict so vLLM/HF exports see only LM weights.
        # The complete projector/optimizer/marginal state is saved alongside every actor checkpoint.
        from verl.utils.emd_bridge import FrozenEMDBridge
        bridge_path = Path(config.emd_method_path).resolve().parents[1] / "artifacts/bridge.pt"
        bridge = FrozenEMDBridge(bridge_path, method["bridge_sha256"], device,
                                expected_rank=method["bridge_rank"])
        self.projector = bridge.student
        self.optimizer = torch.optim.AdamW(self.projector.parameters(), lr=float(config.optim.lr),
                                           weight_decay=float(config.optim.weight_decay))
        self.a = np.full(28, 1 / 28, dtype=np.float64)
        self.b = np.full(36, 1 / 36, dtype=np.float64)
        self.updates = 0
        self.method = method

    def save(self, local_path):
        path = Path(local_path) / "emd_state.pt"
        torch.save(dict(projector=self.projector.state_dict(), optimizer=self.optimizer.state_dict(),
                        a=self.a, b=self.b, updates=self.updates, method=self.method), path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)

    def load(self, local_path):
        state = torch.load(Path(local_path) / "emd_state.pt", map_location="cpu", weights_only=False)
        if state["method"] != self.method:
            raise ValueError("EMD method configuration differs from checkpoint")
        self.projector.load_state_dict(state["projector"])
        self.optimizer.load_state_dict(state["optimizer"])
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
    metrics = {}

    def metric(key, value):
        metrics.setdefault(key, []).append(float(value))

    def forward(micro):
        inputs = {**micro.batch, **micro.non_tensor_batch}
        _, logp, _, _, hs = actor._forward_micro_batch(
            inputs, temperature=temperature, calculate_entropy=False,
            return_response_hidden_repr=True,
            rep_distillation_positions="all", rep_distillation_layers="all")
        from verl.utils.emd_teacher_cache import KEY, load
        teacher = load(inputs[KEY], device) if KEY in inputs else inputs["teacher_last_hidden_repr"]
        cost = layer_cost(hs, teacher, inputs["response_mask"], state.projector)
        return inputs, logp, cost

    for mini in data.split(actor.config.ppo_mini_batch_size):
        microbatches = mini.split(actor.config.ppo_micro_batch_size_per_gpu)
        total = len(mini)
        if total != 32:
            raise ValueError("EMD global optimizer minibatch must contain 32 trajectories")
        costs, rngs = [], []
        # First pass: batch-global costs, no retained model graph and no parameter update.
        with torch.no_grad():
            for micro in microbatches:
                rngs.append((torch.get_rng_state(), torch.cuda.get_rng_state()))
                micro.to(device)
                inputs, logp, cost = forward(micro)
                costs.append(cost.detach().cpu() * len(micro) / total)
                micro.to("cpu")
                del inputs, logp, cost
        global_cost = torch.stack(costs).sum(0)
        flow = exact_flow(global_cost, state.a, state.b).to(device)
        actor.actor_optimizer.zero_grad(set_to_none=True)
        state.optimizer.zero_grad(set_to_none=True)
        observed = torch.zeros_like(global_cost)
        for micro, rng in zip(microbatches, rngs):
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1])
            micro.to(device)
            inputs, logp, cost = forward(micro)
            pg, pg_metrics = compute_policy_loss_vanilla(
                old_log_prob=inputs["old_log_probs"], log_prob=logp,
                advantages=inputs["advantages"].detach(), response_mask=inputs["response_mask"],
                loss_agg_mode="seq-mean-token-mean", config=actor.config)
            emd = (flow * cost).sum()
            representation_only = bool(actor.config.rep_distillation_only)
            loss = combine_objective(pg, emd, state.coefficient, representation_only)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite combined OPD/EMD loss")
            scale = len(micro) / total
            (loss * scale).backward()
            observed += cost.detach().cpu() * scale
            metric("actor/pg_loss", pg.detach().item() * scale)
            metric("actor/emd_loss", emd.detach().item() * scale)
            metric("emd/representation_only", float(representation_only) * scale)
            for key, value in pg_metrics.items(): metric(key, value)
            micro.to("cpu")
            del inputs, logp, cost, pg, emd, loss
        if not torch.allclose(observed, global_cost, rtol=2e-4, atol=2e-5):
            raise RuntimeError("EMD two-pass cost mismatch; parameters/RNG or token alignment changed")
        # One global clip over both trainable components; NO_SHARD FSDP on one GPU.
        student_norm = actor.actor_module.clip_grad_norm_(float("inf"))
        projector_norm = torch.nn.utils.clip_grad_norm_(state.projector.parameters(), float("inf"))
        norm = torch.sqrt(student_norm.float().square() + projector_norm.float().square())
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite student/projector joint gradient")
        coefficient = (float(actor.config.grad_clip) / (norm + 1e-6)).clamp(max=1.)
        for param in list(actor.actor_module.parameters()) + list(state.projector.parameters()):
            if param.grad is not None: param.grad.mul_(coefficient)
        actor_group = actor.actor_optimizer.param_groups[0]
        for group in state.optimizer.param_groups:
            for key in ("lr", "betas", "weight_decay", "eps"):
                group[key] = actor_group[key]
        actor.actor_optimizer.step()
        state.optimizer.step()
        state.a, state.b = update_marginals(global_cost, flow, state.a, state.b, state.tau)
        state.updates += 1
        metric("actor/grad_norm", norm.item())
        metric("emd/projector_grad_norm", projector_norm.item())
        metric("emd/optimizer_updates", state.updates)
    actor.actor_optimizer.zero_grad(set_to_none=True)
    state.optimizer.zero_grad(set_to_none=True)
    return metrics
