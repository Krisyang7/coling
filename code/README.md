# Implementation snapshot

`verl_extensions/` contains the exact files used by the active experiments:

- `utils/bert_emd.py`: layer cost, exact balanced transport, and marginal updates
- `utils/emd_bridge.py`: frozen shared-coordinate Bridge
- `utils/emd_teacher_cache.py`: detached teacher representation cache
- `workers/actor/emd_update.py`: two-pass global mini-batch EMD update
- `workers/actor/dp_actor.py` and `workers/fsdp_workers.py`: actor/teacher integration
- supporting representation, attention, OPD, and capacity utilities required by those integration files

These files are an overlay for the VERL tree distributed with the OPRD implementation. The repository does not vendor model weights, training data, or the complete third-party framework.

The stage-2 configuration disables representation distillation. This lets the same integration tree execute the original sampled-token OPD objective without adding EMD to its loss.
