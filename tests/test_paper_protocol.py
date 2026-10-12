import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_calibration_and_bridge_protocol_matches_paper() -> None:
    calibration = (ROOT / "scripts/calibrate_centered_affine_bridge.py").read_text(encoding="utf-8")
    bridge_runtime = (ROOT / "src/verl/utils/emd_bridge.py").read_text(encoding="utf-8")

    for path in (
        ROOT / "scripts/calibrate_centered_affine_bridge.py",
        ROOT / "src/verl/utils/emd_bridge.py",
    ):
        parse(path)
    assert '"calibration_fit_examples": 1800' in calibration
    assert '"calibration_validation_examples": 200' in calibration
    assert '"calibration_positions_per_example": 64' in calibration
    assert "min(response_length, 2000)" not in calibration
    assert "torch.linalg.lstsq" in calibration
    assert 'parser.add_argument("--seed", type=int, required=True)' in calibration
    assert 'state["rank"] == expected_rank == 8' in bridge_runtime
    assert 'state["calibration_fit_examples"] == 1800' in bridge_runtime


def test_stage_boundary_dispatches_only_stage_two_to_emd() -> None:
    actor = (ROOT / "src/verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
    workers = (ROOT / "src/verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/train.py").read_text(encoding="utf-8")

    assert 'if self.config.get("use_rep_distillation", False):' in actor
    assert 'if self.emd_state is not None:' in actor
    assert "return update_policy(self, data)" in actor
    assert workers.count("if self.actor.emd_state is not None:") == 2
    assert 'environment["EMD_BRIDGE_METHOD_PATH"]' in launcher


def test_undocumented_training_values_remain_explicit_placeholders() -> None:
    for name in ("emd_train.json", "opd_train.json"):
        config = json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))
        assert config["_paper"]["training_problems"] == 7500
        assert any(isinstance(value, str) and value.startswith("<SET_") for value in config.values())
