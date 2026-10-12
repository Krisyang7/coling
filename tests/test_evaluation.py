import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("evaluate", ROOT / "scripts/evaluate.py")
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_avg8_and_pass8_summary(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    rows = []
    for question_id, correct_count in (("q0", 2), ("q1", 0)):
        rows.extend(
            {
                "task": "toy",
                "question_id": question_id,
                "rollout_id": rollout_id,
                "correct": rollout_id < correct_count,
                "response_tokens": 1,
            }
            for rollout_id in range(8)
        )
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = MODULE.summarize(output, "toy", n=8)

    assert summary["questions"] == 2
    assert summary["rollouts"] == 16
    assert summary["avg_at_8"] == 0.125
    assert summary["pass_at_8"] == 0.5


def test_evaluation_defaults_match_paper() -> None:
    source = (ROOT / "scripts/evaluate.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--seed", type=int, required=True)' in source
    assert 'parser.add_argument("--n", type=int, default=8)' in source
    assert 'parser.add_argument("--max-tokens", type=int, default=8192)' in source
    assert 'parser.add_argument("--thinking-mode", choices=("enabled", "disabled"), required=True)' in source
    assert 'parser.add_argument("--max-model-len", type=int, required=True)' in source
    assert "add_answer_instruction" not in source
    assert "Please reason step by step" not in source
    assert "temperature=1.0" in source
    assert "top_p=0.8" in source
