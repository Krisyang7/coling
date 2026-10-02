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
"""Frozen rank-8/32/64 bridge in ONE teacher coordinate system for cross-layer EMD.

This is a new method variant, not lossless compression of full-space BERT-EMD.
OPRD's independent per-layer PCA coordinates must not be compared across layers.
"""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StudentBridge(torch.nn.Module):
    def __init__(self, ps):
        super().__init__()
        self.weight = torch.nn.Parameter(ps.detach().clone(), requires_grad=False)

    def forward(self, hidden):
        # layer_cost calls this with [28, valid_response_tokens, 2048].
        return torch.einsum("lkd,lrd->lkr", hidden.float(), self.weight)


class FrozenEMDBridge:
    def __init__(self, path, expected_sha256, device, expected_rank=8):
        if not expected_sha256 or sha256(path) != expected_sha256:
            raise ValueError("Missing or changed shared EMD bridge hash")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not (state["complete"] and state["kind"] == "shared_teacher_pca_emd_v1"
                 and state["rank"] in (8, 32, 64) and state["rank"] == expected_rank
                and state["calibration_prompts"] == 2000
                and state["epochs"] == 20 and state["provenance"]):
            raise ValueError("Incomplete or incompatible shared EMD bridge")
        self.rank = state["rank"]
        for key, shape in {"ps": (28, self.rank, 2048), "pt": (self.rank, 4096), "mean": (4096,)}.items():
            value = state[key]
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid shared EMD bridge tensor: {key}")
            setattr(self, key, value.to(device=device, dtype=torch.float32).detach())
        self.student = StudentBridge(self.ps).to(device)

    @torch.no_grad()
    def teacher(self, hidden):
        if hidden.ndim != 4 or hidden.shape[1] != 36 or hidden.shape[-1] != 4096:
            raise ValueError("Shared EMD bridge needs [batch,36,response,4096]")
        output = torch.empty((*hidden.shape[:-1], self.rank), device=hidden.device, dtype=torch.bfloat16)
        # Project before copying to host; bounded FP32 temporaries, all layers/tokens.
        for layer in range(36):
            for start in range(0, hidden.shape[2], 256):
                value = hidden[:, layer, start:start + 256].float()
                output[:, layer, start:start + 256] = F.linear(value - self.mean, self.pt)
        return output
