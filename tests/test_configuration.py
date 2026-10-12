import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage_two_uses_only_fixed_uniform_emd() -> None:
    method = json.loads((ROOT / "config/method.json").read_text(encoding="utf-8"))
    train = json.loads((ROOT / "config/emd_train.json").read_text(encoding="utf-8"))
    source = (ROOT / "src/verl/workers/actor/emd_update.py").read_text(encoding="utf-8")
    solver = (ROOT / "src/verl/utils/bert_emd.py").read_text(encoding="utf-8")

    assert method["marginals"] == "uniform_fixed"
    assert method["transport"] == "exact_unregularized_balanced_lp"
    assert method["mass_quantization_scale"] == 10**12
    assert method["representation_epsilon"] == 1e-12
    assert method["bridge_kind"] == "shared_teacher_pca_centered_affine_emd_v1"
    assert method["bridge_rank"] == 8
    assert method["calibration_fit_examples"] == 1800
    assert method["calibration_validation_examples"] == 200
    assert method["calibration_positions_per_example"] == 64
    assert method["training_problems"] == 7500
    assert method["lambda_emd"] is None
    assert method["solver"]
    assert all(value is None for value in method["solver"].values())
    assert method["objective_schedule"] == [
        "stage_2: hidden_emd_loss_only",
        "stage_3: sampled_token_opd_loss_only",
    ]
    assert train["+actor_rollout_ref.actor.use_rep_distillation"] is True
    assert train["+actor_rollout_ref.actor.rep_distillation_only"] is True
    assert train["+actor_rollout_ref.actor.rep_distillation_positions"] == "last_k"
    assert train["+actor_rollout_ref.actor.rep_distillation_last_k"] == 2000
    assert train["actor_rollout_ref.actor.ppo_epochs"] == 1
    assert train["trainer.n_gpus_per_node"] == 1
    assert train["trainer.nnodes"] == 1
    assert train["+data.apply_chat_template_kwargs.enable_thinking"] == "<SET_STAGE_II_ENABLE_THINKING>"
    assert "tau" not in method
    assert "update_marginals(" not in source
    assert "update_marginals(" not in solver
    assert "state.optimizer" not in source
    assert "compute_policy_loss" not in source
    assert "must contain 32 trajectories" not in source
    assert 'metric("emd/uniform_marginals", 1.0)' in source
    assert "MASS_QUANTIZATION_SCALE = 10**12" in solver


def test_stage_three_disables_representation_distillation() -> None:
    opd = json.loads((ROOT / "config/opd_train.json").read_text(encoding="utf-8"))

    assert opd["+actor_rollout_ref.actor.use_rep_distillation"] is False
    assert opd["actor_rollout_ref.actor.optim.optimizer"] == "AdamW"
    assert opd["actor_rollout_ref.actor.optim.lr_scheduler_type"] == "cosine"
    assert opd["actor_rollout_ref.actor.clip_ratio"] == 0.2
    assert opd["actor_rollout_ref.actor.loss_agg_mode"] == "seq-mean-token-mean"
    assert opd["actor_rollout_ref.actor.use_kl_loss"] is False
    assert opd["actor_rollout_ref.actor.entropy_coeff"] == 0.0
    assert opd["trainer.resume_mode"] == "disable"
    assert opd["+data.apply_chat_template_kwargs.enable_thinking"] == "<SET_STAGE_III_ENABLE_THINKING>"
    assert not any("emd" in key.lower() for key in opd)
