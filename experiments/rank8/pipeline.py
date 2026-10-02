"""One-shot rank-8 Bridge-EMD -> sampled-token OPD experiment on GPU 6."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
RUN_ID = os.environ["EXPERIMENT_RUN_ID"]
RUN = EXP / "runs" / RUN_ID
STATE = RUN / "state.json"
PYTHON = ROOT / ".venv/bin/python"
GPU = "6"


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def write_state(status, **extra):
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state.update(status=status, updated_at=time.time(), run_id=RUN_ID, gpu=int(GPU), **extra)
    write_json(STATE, state)


def gpu_snapshot():
    values = subprocess.check_output([
        "nvidia-smi", "-i", GPU,
        "--query-gpu=memory.used,memory.total,utilization.gpu,uuid",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    processes = subprocess.check_output([
        "nvidia-smi", "-i", GPU,
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    return {
        "used_mib": int(values[0].strip()),
        "total_mib": int(values[1].strip()),
        "util": int(values[2].strip()),
        "uuid": values[3].strip(),
        "processes": sorted(row.strip() for row in processes if row.strip()),
    }


def wait_gpu(stage):
    stable_since = None
    stable_processes = None
    while True:
        snapshot = gpu_snapshot()
        admissible = snapshot["used_mib"] <= 6144 and snapshot["util"] == 0
        if admissible and snapshot["processes"] == stable_processes:
            stable_since = stable_since or time.time()
        elif admissible:
            stable_processes = snapshot["processes"]
            stable_since = time.time()
        else:
            stable_processes = None
            stable_since = None
        stable_seconds = 0 if stable_since is None else int(time.time() - stable_since)
        write_state("waiting_gpu6", pending_stage=stage, gpu_snapshot=snapshot,
                    stable_seconds=stable_seconds)
        if stable_seconds >= 60:
            return snapshot
        time.sleep(5)


def stage_env(base, stage, snapshot, required_mib):
    admitted = []
    for row in snapshot["processes"]:
        pid_text, used_text = row.split(",", 1)
        pid = int(pid_text.strip())
        proc = Path(f"/proc/{pid}")
        try:
            uid = proc.stat().st_uid
            start = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        except (FileNotFoundError, ProcessLookupError):
            continue
        admitted.append({"pid": pid, "uid": uid, "start": start,
                         "used_mib": int(used_text.strip())})
    env = base.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=GPU,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        EXPECTED_GPU_UUID=snapshot["uuid"],
        EXPERIMENT_REQUIRED_FREE_MIB=str(required_mib),
        EXPERIMENT_STAGE_PGID=str(os.getpgrp()),
        EXPERIMENT_ADMITTED_PROCESSES=json.dumps(admitted),
        EXPERIMENT_RUNTIME_STATUS=str(RUN / f"{stage}_runtime.json"),
        EXPERIMENT_RUN_ID=RUN_ID,
        EXPERIMENT_PIPELINE_STAGE=stage,
    )
    return env


def run_logged(stage, command, env, cwd=ROOT):
    log = RUN / "logs" / f"{stage}.log"
    if log.exists():
        raise RuntimeError(f"refusing to overwrite {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    write_state(f"running_{stage}", stage=stage, log=str(log))
    with log.open("w") as stream:
        subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def run_gpu_stage(stage, launcher, smoke, required_mib):
    snapshot = wait_gpu(stage)
    env = stage_env(os.environ, stage, snapshot, required_mib)
    command = [str(PYTHON), str(EXP / "launchers" / launcher), "--run-dir", str(RUN)]
    if smoke:
        command.append("--smoke")
    run_logged(stage, command, env)


def verify_smoke(kind, required, forbidden=None):
    checkpoint = RUN / "checkpoints" / kind / "global_step_1"
    metrics = RUN / "logs" / kind / "train_metrics.log"
    if not checkpoint.is_dir() or not metrics.is_file():
        raise RuntimeError(f"{kind} did not produce a complete one-step checkpoint and metrics")
    text = metrics.read_text(errors="replace")
    if required not in text or (forbidden and forbidden in text):
        raise RuntimeError(f"{kind} objective witness failed")
    marker = {
        "status": "passed", "stage": kind, "verified_at": time.time(),
        "required_metric": required, "forbidden_metric": forbidden,
        "checkpoint_removed_after_verification": True,
    }
    write_json(RUN / f"{kind}_verified.json", marker)
    shutil.rmtree(RUN / "checkpoints" / kind)


def merge(stage, actor, target, vendor):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(EXP / vendor)
    run_logged(stage, [str(PYTHON), "-m", "verl.model_merger", "merge",
                        "--backend", "fsdp", "--local_dir", str(actor),
                        "--target_dir", str(target)], env, EXP / vendor)


def main():
    if RUN.exists():
        raise RuntimeError(f"run already exists: {RUN}")
    if shutil.disk_usage(EXP).free < 60 * 1024**3:
        raise RuntimeError("sequential run requires at least 60 GiB free disk")
    import psutil
    if psutil.virtual_memory().available < 640 * 1024**3:
        raise RuntimeError("sequential EMD retains the validated 640 GiB host-RAM gate")

    RUN.mkdir(parents=True)
    write_state("initializing")
    method = json.loads((EXP / "config/method.json").read_text())
    write_json(RUN / "manifest.json", {
        "run_id": RUN_ID,
        "method": "rank-8 frozen Bridge, EMD-only, then sampled-token OPD-only",
        "objective_schedule": ["EMD-only:174", "OPD-only:174"],
        "bridge_sha256": method["bridge_sha256"],
        "shared_protocol": {"batch": 128, "mini_batch": 32, "micro_batch": 1,
                            "epochs_per_stage": 3, "steps_per_stage": 174,
                            "seed": 42, "response_length": 4096},
        "gpu": int(GPU),
    })

    run_gpu_stage("emd_smoke", "emd_train.py", True, 68869)
    verify_smoke("emd_smoke", "emd/representation_only")
    run_gpu_stage("emd_train", "emd_train.py", False, 68869)
    emd_actor = RUN / "checkpoints/emd_train/global_step_174/actor"
    if not emd_actor.is_dir():
        raise RuntimeError("EMD-only training exited without global_step_174")
    merge("emd_merge", emd_actor, RUN / "emd_final_hf", "vendor_emd/verl")

    run_gpu_stage("opd_smoke", "opd_train.py", True, 61440)
    verify_smoke("opd_smoke", "actor/pg_loss", "actor/emd_loss")
    run_gpu_stage("opd_train", "opd_train.py", False, 61440)
    opd_actor = RUN / "checkpoints/opd_train/global_step_174/actor"
    if not opd_actor.is_dir():
        raise RuntimeError("OPD-only training exited without global_step_174")
    final_hf = RUN / "final_hf"
    merge("opd_merge", opd_actor, final_hf, "vendor_opd")

    results = RUN / "results"
    results.mkdir()
    eval_env = os.environ.copy()
    eval_env["PYTHONPATH"] = str(EXP / "vendor_opd")
    evaluator = EXP / "launchers/evaluate.py"
    for task, folder in (("MATH500", "MATH-500"), ("AMC23", "AMC23"), ("AIME24", "AIME24")):
        snapshot = wait_gpu(task)
        env = stage_env(eval_env, task, snapshot, 60000)
        run_logged(task, [str(PYTHON), str(evaluator), "--model", str(final_hf),
                           "--task", task,
                           "--data", str(ROOT / "third_party/OPRD/datasets/test_data" / folder / "test.parquet"),
                           "--output", str(results / f"{task}.jsonl"),
                           "--summary", str(results / f"{task}_summary.json"),
                           "--seed", "42", "--n", "8", "--max-tokens", "8192"], env)
    summaries = {task: json.loads((results / f"{task}_summary.json").read_text())
                 for task in ("MATH500", "AMC23", "AIME24")}
    write_state("completed", stage="completed", final_hf=str(final_hf), results=summaries)


if __name__ == "__main__":
    main()
