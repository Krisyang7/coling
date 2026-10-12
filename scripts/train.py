from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def hydra_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def placeholders(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            found.extend(placeholders(item, f"{prefix}.{key}" if prefix else key))
        return found
    if isinstance(value, list):
        found = []
        for index, item in enumerate(value):
            found.extend(placeholders(item, f"{prefix}[{index}]"))
        return found
    if isinstance(value, str) and value.startswith("<SET_"):
        return [prefix]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    missing = placeholders(config)
    if missing:
        raise ValueError("replace required config placeholders: " + ", ".join(missing))
    overrides = [
        f"{key}={hydra_value(value)}"
        for key, value in config.items()
        if not key.startswith("_")
    ]
    environment = os.environ.copy()
    method_path = config.get("+actor_rollout_ref.actor.emd_method_path")
    if method_path:
        environment["EMD_BRIDGE_METHOD_PATH"] = str(Path(method_path).resolve())
    subprocess.run(
        [sys.executable, "-m", "verl.trainer.main_ppo", *overrides],
        check=True,
        env=environment,
    )


if __name__ == "__main__":
    main()
