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
"""Lossless teacher cache for the single-process, colocated EMD worker.

Only opaque row keys travel through Ray. Each BF16 teacher row stays in CPU
memory in its producing worker until that batch's actor update finishes.
"""
import os
import uuid

import numpy as np
import torch

KEY = "emd_teacher_cache_keys"
_ROWS = {}


def store(hidden):
    if hidden.ndim != 4:
        raise ValueError("Expected [batch, layers, response, hidden] teacher states")
    keys = []
    for row in hidden.split(1):
        key = f"{os.getpid()}:{uuid.uuid4().hex}"
        # Same BF16 conversion as the previous full-batch CPU cache. Own the
        # storage so a slice cannot retain a larger GPU/CPU allocation.
        _ROWS[key] = row.detach().to(device="cpu", dtype=torch.bfloat16, copy=True).contiguous()
        keys.append(key)
    return np.asarray(keys, dtype=object)


def load(keys, device):
    if len(keys) == 0:
        raise ValueError("Empty teacher cache selection")
    rows = []
    for key in keys:
        if not str(key).startswith(f"{os.getpid()}:") or key not in _ROWS:
            raise RuntimeError("EMD teacher cache missing: teacher and actor must share one worker process")
        rows.append(_ROWS[key].to(device=device))
    return rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)


def release(keys):
    for key in keys:
        _ROWS.pop(key, None)


def cached_bytes():
    return sum(row.numel() * row.element_size() for row in _ROWS.values())
