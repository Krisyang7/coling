"""Single-GPU capacity checks between updates and before vLLM cache wakeup."""
import csv
import io
import json
import os
import subprocess
import time
from pathlib import Path

_external = None


def query(fields, kind):
    result = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=15)
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def read_sample(uuid, pgid):
    rows = [r for r in query("uuid,memory.free,utilization.gpu", "gpu") if r[0] == uuid]
    if len(rows) != 1:
        raise RuntimeError("Expected GPU UUID missing from telemetry")
    owned, external, owned_processes = 0, [], []
    for gpu, pid, used in query("gpu_uuid,pid,used_gpu_memory", "compute-apps"):
        if gpu != uuid:
            continue
        pid = int(pid)
        proc = Path(f"/proc/{pid}")
        uid = proc.stat().st_uid
        start = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        if uid == os.getuid() and (pid == os.getpid() or os.getpgid(pid) == pgid):
            owned += int(used)
            owned_processes.append((pid, uid, start))
        else:
            external.append((pid, uid, start))
    return dict(capacity_mib=int(rows[0][1]) + owned, free_mib=int(rows[0][1]),
                owned_mib=owned, owned_processes=owned_processes,
                util=int(rows[0][2]), external=sorted(external))


def write_status(path, payload):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    tmp.replace(path)


def wait_for_capacity(phase, min_free_mib=0):
    """Never alter other processes; retain the current batch/weights while waiting.

    Count only this stage's resident memory as reclaimable capacity. After a
    shortage, telemetry failure, or changed external process set, require a new
    stable 60-second low-load window with the same admitted external processes. This is
    observation, not a GPU reservation.
    """
    global _external
    uuid = os.environ["EXPECTED_GPU_UUID"]
    required = int(os.environ["EXPERIMENT_REQUIRED_FREE_MIB"])
    pgid = int(os.environ["EXPERIMENT_STAGE_PGID"])
    path = Path(os.environ["EXPERIMENT_RUNTIME_STATUS"])
    if _external is None:
        _external = sorted((p["pid"], p["uid"], p["start"])
                           for p in json.loads(os.environ["EXPERIMENT_ADMITTED_PROCESSES"]))
    blocked, history = False, []
    while True:
        now = time.monotonic()
        try:
            sample = read_sample(uuid, pgid)
            enough = (sample["capacity_mib"] >= required
                      and sample["free_mib"] >= min_free_mib)
            same_external = sample["external"] == _external
            if not blocked and enough and same_external:
                write_status(path, dict(status="running", phase=phase, updated_at=time.time(),
                                        required_mib=required, min_free_mib=min_free_mib, **sample))
                return
            blocked = True
            if not enough or sample["util"] > 50:
                history = []
            else:
                if history and (now - history[-1][0] > 12.5 or
                                sample["external"] != history[-1][1]["external"]):
                    history = []
                history.append((now, sample))
                while len(history) > 1 and history[1][0] <= now - 60:
                    history.pop(0)
                if (now - history[0][0] >= 60 and
                        sum(s["util"] for _, s in history) / len(history) <= 30 and
                        max(s["capacity_mib"] for _, s in history) -
                        min(s["capacity_mib"] for _, s in history) <= 2048):
                    _external = sample["external"]
                    write_status(path, dict(status="running", phase=phase, updated_at=time.time(),
                                            required_mib=required, min_free_mib=min_free_mib, **sample))
                    return
            payload = dict(status="waiting", phase=phase, updated_at=time.time(),
                           required_mib=required, min_free_mib=min_free_mib, **sample)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            blocked, history = True, []
            payload = dict(status="waiting", phase=phase, updated_at=time.time(),
                           required_mib=required, telemetry_error=str(exc))
        write_status(path, payload)
        print("RUNTIME_CAPACITY_WAIT", json.dumps(payload), flush=True)
        time.sleep(5)
