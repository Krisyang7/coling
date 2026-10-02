"""Stage 2: continue from the EMD checkpoint with sampled-token OPD only."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    kind = "opd_smoke" if args.smoke else "opd_train"
    output = args.run_dir / "checkpoints" / kind
    logs = args.run_dir / "logs" / kind
    actor = args.run_dir / "emd_final_hf"
    if not (actor / "model.safetensors").is_file():
        raise RuntimeError("OPD stage requires the merged EMD-only model")
    if output.exists() or logs.exists():
        raise RuntimeError(f"refusing to overwrite existing {kind} attempt")

    config = json.loads((EXP / "config/opd_train.json").read_text())
    if args.smoke:
        config.update(json.loads((EXP / "config/smoke.json").read_text()))
    config.update({
        "actor_rollout_ref.model.path": str(actor),
        "trainer.default_local_dir": str(output),
        "trainer.output_log_path": str(logs / "train_metrics.log"),
        "trainer.experiment_name": "bert_emd_bridge_rank8_then_opd",
        "trainer.save_freq": 1 if args.smoke else 174,
        "trainer.resume_mode": "disable",
        "+ray_kwargs.ray_init.address": "local",
        "+ray_kwargs.ray_init._temp_dir": "/tmp/yo-" + hashlib.sha256(
            f"{args.run_dir}/{kind}".encode()
        ).hexdigest()[:10],
        "+ray_kwargs.ray_init.include_dashboard": False,
        "ray_kwargs.ray_init.num_cpus": 8,
    })
    if config.get("+actor_rollout_ref.actor.use_rep_distillation") is not False:
        raise RuntimeError("OPD stage must disable representation distillation")
    if any("emd" in key.lower() for key in config):
        raise RuntimeError("OPD stage contains an EMD-specific configuration key")

    logs.mkdir(parents=True)
    (logs / "launch_config.json").write_text(json.dumps(config, indent=2) + "\n")
    command = [sys.executable, "-m", "verl.trainer.main_ppo"] + [
        f"{key}={render(value)}" for key, value in config.items()
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(EXP / "vendor_opd")
    env.pop("RAY_ADDRESS", None)
    subprocess.run(command, cwd=EXP / "vendor_opd", env=env, check=True)


if __name__ == "__main__":
    main()
