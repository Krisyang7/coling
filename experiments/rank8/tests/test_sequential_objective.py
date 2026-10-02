import json
from pathlib import Path

import torch

from verl.workers.actor.emd_update import combine_objective


EXP = Path(__file__).resolve().parents[1]


def test_representation_only_excludes_policy_gradient():
    pg = torch.tensor(2.0, requires_grad=True)
    emd = torch.tensor(3.0, requires_grad=True)
    combine_objective(pg, emd, 1.5, True).backward()
    assert pg.grad is None
    assert emd.grad.item() == 1.5


def test_stage_configs_are_mutually_exclusive():
    emd = json.loads((EXP / "config/emd_train.json").read_text())
    opd = json.loads((EXP / "config/opd_train.json").read_text())
    assert emd["+actor_rollout_ref.actor.use_rep_distillation"] is True
    assert emd["+actor_rollout_ref.actor.rep_distillation_only"] is True
    assert opd["+actor_rollout_ref.actor.use_rep_distillation"] is False
    assert not any("emd" in key.lower() for key in opd)
    shared = set(emd) & set(opd)
    allowed = {
        "+actor_rollout_ref.actor.use_rep_distillation",
        "trainer.experiment_name",
        "trainer.save_freq",
    }
    assert all(emd[key] == opd[key] for key in shared - allowed)
