# Experiment status

This is a point-in-time record captured on **2026-10-02 11:51 CST**. It distinguishes completed objectives from running stages and does not treat a checkpoint or smoke test as a final result.

| Experiment | Completed work | Active stage | Progress | Latest durable checkpoint | Evaluation |
|---|---|---|---:|---|---|
| Bridge-EMD → OPD, rank 8 | EMD smoke, EMD 174/174, EMD merge, OPD smoke | sampled-token OPD | 37/174 | EMD `global_step_174`; OPD `global_step_1` | pending |
| Bridge-EMD → OPD, rank 32 | EMD smoke, EMD 174/174, EMD merge, OPD smoke | sampled-token OPD | 19/174 | EMD `global_step_174`; OPD `global_step_1` | pending |
| Bridge-EMD → OPD, rank 64 | EMD smoke | EMD-only representation distillation | 125/174 | none yet; configured to save at step 174 | pending |
| OPRD-Bridge → OPD | Bridge initialization, OPD 174/174, merge | complete | 174/174 | `global_step_174` | MATH500, AMC23, AIME24 complete |

All three active EMD experiments use the same teacher, student, data object, prompt, training hyperparameters, and evaluation protocol. Only the frozen Bridge rank differs.

The rank-8 and rank-32 OPD stages explicitly set `use_rep_distillation=false`. Their extra EMD machinery is not part of the second-stage loss.
