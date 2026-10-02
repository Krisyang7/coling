"""Stage 1: train the student with the Bridge-EMD objective only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]


def render(value):
    if isinstance(value, str):
        return value.replace("{root}", str(ROOT)).replace("{experiment}", str(EXP))
    return json.dumps(value)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    kind = "emd_smoke" if args.smoke else "emd_train"
    output = args.run_dir / "checkpoints" / kind
    logs = args.run_dir / "logs" / kind
    if output.exists() or logs.exists():
        raise RuntimeError(f"refusing to overwrite existing {kind} attempt")

    config = json.loads((EXP / "config/emd_train.json").read_text())
    if args.smoke:
        config.update(json.loads((EXP / "config/smoke.json").read_text()))
    config.update({
        "trainer.default_local_dir": str(output),
        "trainer.output_log_path": str(logs / "train_metrics.log"),
        "trainer.experiment_name": "bert_emd_bridge_rank8_emd_only",
        "trainer.save_freq": 1 if args.smoke else 174,
        "trainer.resume_mode": "disable",
        "+ray_kwargs.ray_init.address": "local",
        "+ray_kwargs.ray_init._temp_dir": "/tmp/yo-" + hashlib.sha256(
            f"{args.run_dir}/{kind}".encode()
        ).hexdigest()[:10],
        "+ray_kwargs.ray_init.include_dashboard": False,
        "ray_kwargs.ray_init.num_cpus": 8,
        "+ray_kwargs.ray_init.runtime_env.env_vars.EMD_BRIDGE_METHOD_PATH": str(
            EXP / "config/method.json"
        ),
    })
    if config.get("+actor_rollout_ref.actor.use_rep_distillation") is not True:
        raise RuntimeError("EMD stage must enable representation distillation")
    if config.get("+actor_rollout_ref.actor.rep_distillation_only") is not True:
        raise RuntimeError("EMD stage must exclude the OPD gradient")

    method = json.loads((EXP / "config/method.json").read_text())
    actual_bridge = sha256(EXP / "artifacts/bridge.pt")
    if actual_bridge != method.get("bridge_sha256"):
        raise RuntimeError("rank-8 Bridge SHA-256 mismatch")

    logs.mkdir(parents=True)
    (logs / "launch_config.json").write_text(json.dumps(config, indent=2) + "\n")
    command = [sys.executable, "-m", "verl.trainer.main_ppo"] + [
        f"{key}={render(value)}" for key, value in config.items()
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(EXP / "vendor_emd/verl")
    env["EMD_BRIDGE_METHOD_PATH"] = str(EXP / "config/method.json")
    env.pop("RAY_ADDRESS", None)
    subprocess.run(command, cwd=EXP / "vendor_emd/verl", env=env, check=True)


if __name__ == "__main__":
    main()
