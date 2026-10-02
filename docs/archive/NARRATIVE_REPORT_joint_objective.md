# Narrative Report: EMD-OPD

## Framing update: OPD versus OPRD versus our method

The paper's primary comparison is OPD, the current representation-only OPRD-Bridge baseline, and EMD-OPD. EOPD is a reference for manuscript organization, experimental methodology, and common settings; it is not the central research target or a claimed reproduced method.

| Axis | OPD | Current OPRD-Bridge | EMD-OPD |
|---|---|---|---|
| Output objective | Sampled-token OPD | Absent | Same sampled-token OPD |
| Representation objective | Absent | Fixed-pair normalized MSE | Transport-weighted MSE |
| Online layer correspondence | None | Proportional fixed map | Cost-dependent many-to-many EMD |
| Bridge space | None | Pair-specific | Shared teacher coordinates |

The contribution is the tested on-policy adaptation, not the invention of hidden-state distillation, frozen bridges, or EMD. Bridge calibration retains a proportional-map prior. The matched shared-bridge fixed-map+OPD control is necessary to attribute gains specifically to transport.

## Working thesis

On-policy output distillation and hidden-state distillation provide different explicit training signals. The completed controlled baselines show that sampled-token OPD is substantially stronger than the current pure OPRD-Bridge configuration on Qwen3-1.7B mathematical reasoning. This motivates testing whether a representation loss adds value when the successful OPD objective is retained, and whether cost-dependent many-to-many transport improves upon fixed matching under otherwise identical conditions.

## Method idea

**EMD-OPD** augments the existing sampled-token OPD objective with a BERT-EMD-style hidden-state transport loss. The teacher and student process the same student-generated response. A frozen shared low-rank bridge maps their hidden states into one coordinate system. Every student layer can then match every teacher layer through an exact balanced optimal-transport plan over the full layer-distance matrix. The plan is detached; gradients flow only from the transport-weighted costs through the student bridge into the student model.

The proposed objective is

\[
\mathcal{L}_{\text{EMD-OPD}}
=\mathcal{L}_{\text{sampled-OPD}}
+\lambda_{\text{emd}}\sum_{i,j}\operatorname{stopgrad}(F_{ij})C_{ij},
\]

with \(\lambda_{\text{emd}}=1\) and transport temperature \(\tau=1\). The current implementation evaluates bridge ranks 8, 32, and 64.

## Evidence already available

All completed results use Qwen3-8B as teacher, Qwen3-1.7B-Base as student, seed 42, and a frozen evaluation protocol with eight samples per problem.

| Method | MATH500 Avg@8 / Pass@8 | AMC23 Avg@8 / Pass@8 | AIME24 Avg@8 / Pass@8 | Macro Avg@8 / Pass@8 |
|---|---:|---:|---:|---:|
| Student | 4.150 / 28.200 | 2.8125 / 15.000 | 0.4167 / 3.3333 | 2.46 / 15.51 |
| Sampled-token OPD | 62.175 / 84.000 | 34.0625 / 67.500 | 7.0833 / 20.000 | 34.44 / 57.17 |
| OPRD-Bridge, current | 23.525 / 73.600 | 16.5625 / 55.000 | 2.500 / 6.6667 | 14.20 / 45.09 |

The result audit reconstructed all reported scores from the saved rollout files and found no missing or duplicate sample IDs. It did not independently re-grade answers. The OPRD audit found no core implementation defect. A displayed representation-loss value was scaled incorrectly in logs, but the optimized gradient was correct.

## Claims that are currently supportable

1. Under this controlled configuration, sampled-token OPD is the strongest completed baseline.
2. Pure fixed-map OPRD-Bridge improves substantially over the undistilled student but does not outperform sampled-token OPD.
3. The proposed method preserves OPD and replaces fixed one-to-one layer assignment with adaptive many-to-many transport in a shared coordinate system.
4. No effectiveness claim about EMD-OPD is supportable until rank-8/32/64 training and evaluation finish.

## Main experiment still required

The headline comparison is EMD-OPD versus sampled-token OPD and current OPRD-Bridge under the same training and evaluation protocol. Causal attribution requires a matched four-way experiment: OPD-only; shared-bridge fixed-map+OPD; shared-bridge EMD-only; and shared-bridge EMD+OPD. The bridge, rank, response-token mask, cost normalization, loss coefficient, and training budget must be identical across the relevant rows. Rank 8, 32, and 64 then characterize a configuration tradeoff rather than isolating rank alone, because rank also changes geometry and the relative EMD-loss scale.

## Scope and limitations

This is a controlled adaptation of the public EOPD protocol, not a strict reproduction. The local baseline uses a sampled-token OPD approximation and a frozen evaluation prompt shared by all compared methods. Existing results come from one training seed and one fixed evaluation seed, so they support configuration-specific comparisons rather than broad statistical claims.
