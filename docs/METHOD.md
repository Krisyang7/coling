# Sequential Bridge-EMD → OPD

## Question

Sampled-token OPD gives the student direct output supervision on student-generated trajectories. OPRD adds intermediate representation supervision, but fixed layer matching can be brittle when teacher and student depth differ. This study asks whether exact optimal transport across layers can provide a better representation initialization before the unchanged OPD stage.

## Frozen shared Bridge

Teacher and student hidden widths differ. A shared teacher PCA basis maps every selected teacher layer into one rank-`r` coordinate system. A projection is fitted for each student layer into that same coordinate system. The projections are then frozen and SHA-256 locked.

The sweep uses `r in {8, 32, 64}`. Bridge calibration is identical except for rank: 2,000 paired examples, the same 1,800/200 split, 20 fitting epochs, and seed 42.

## Stage 1: EMD-only representation distillation

For each global mini-batch, projected response-token hidden states define a layer cost matrix

```math
C_{ij} = \operatorname{MSE}(P_i^S H_i^S, P^T H_j^T).
```

The teacher tensor is detached. Given positive student and teacher layer marginals `a` and `b`, the balanced transport plan is solved exactly:

```math
F^* = \arg\min_{F \ge 0} \langle F, C \rangle
\quad \text{s.t.} \quad F\mathbf{1}=a,\;F^\top\mathbf{1}=b.
```

`F` is detached. The differentiable loss is recomputed through the student features:

```math
L_{EMD} = \lambda_{EMD} \sum_{i,j} \operatorname{stopgrad}(F^*_{ij}) C_{ij},
```

with `lambda_emd=1.0` and `tau=1.0`. This stage contains no policy-gradient/OPD term. It runs for 174 optimizer steps and saves the final checkpoint.

## Stage 2: sampled-token OPD-only

The stage-1 checkpoint is merged, then a fresh AdamW optimizer and cosine schedule are created. Representation distillation is disabled. The student runs the existing sampled-token OPD objective for another 174 steps under the same EOPD-derived settings used by the OPD baseline.

This stage separation prevents persistent gradient competition between output calibration and representation alignment. It also gives a direct mechanistic comparison to OPRD-Bridge → OPD.

## What the rank sweep can establish

Ranks 8, 32, and 64 test a capacity tradeoff in the shared representation space. They do not isolate rank alone perfectly: rank also changes projected geometry and loss scale. All three final evaluations must therefore be reported, without selecting a rank after looking at test performance.

The primary endpoint is macro Avg@8 against sampled-token OPD. Pass@8, response length, truncation, peak memory, and wall time are secondary diagnostics.

## Controls still needed for a transport-specific claim

If the sequential Bridge-EMD method improves over OPD, attributing the gain to adaptive transport requires at least:

1. OPD-only with the same total training budget.
2. Shared-Bridge fixed layer map → OPD.
3. Shared-Bridge EMD → OPD.
4. A uniform or frozen transport-plan control if fixed-map and EMD differ.

Until those controls finish, the supported claim is about the complete sequential initialization method, not EMD routing in isolation.
