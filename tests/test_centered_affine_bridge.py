from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/verl/utils/emd_bridge.py"
SPEC = spec_from_file_location("emd_bridge", MODULE_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
StudentBridge = MODULE.StudentBridge


def test_student_bridge_applies_centering_and_teacher_offset() -> None:
    ps = torch.zeros(2, 3, 4)
    ps[:, :, :3] = torch.eye(3)[None]
    student_mean = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    target_mean = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    hidden = student_mean[:, None, :].repeat(1, 5, 1)
    hidden[:, :, :3] += 2.0

    bridge = StudentBridge(ps, student_mean, target_mean)
    actual = bridge(hidden)
    expected = target_mean[:, None, :].expand(-1, 5, -1) + 2.0

    torch.testing.assert_close(actual, expected)
    assert not any(parameter.requires_grad for parameter in bridge.parameters())
