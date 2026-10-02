# Paper Plan: EMD-OPD

## Framing update: OPD versus OPRD versus our method

The paper's primary comparison is OPD, the current representation-only OPRD-Bridge baseline, and EMD-OPD. EOPD is a reference for manuscript organization, experimental methodology, and common settings; it is not the central research target or a claimed reproduced method.

| Axis | OPD | Current OPRD-Bridge | EMD-OPD |
|---|---|---|---|
| Output objective | Sampled-token OPD | Absent | Same sampled-token OPD |
| Representation objective | Absent | Fixed-pair normalized MSE | Transport-weighted MSE |
| Online layer correspondence | None | Proportional fixed map | Cost-dependent many-to-many EMD |
| Bridge space | None | Pair-specific | Shared teacher coordinates |

The contribution is the tested on-policy adaptation, not the invention of hidden-state distillation, frozen bridges, or EMD. Bridge calibration retains a proportional-map prior. The matched shared-bridge fixed-map+OPD control is necessary to attribute gains specifically to transport.

## One-sentence contribution

We introduce EMD-OPD, an on-policy distillation objective that preserves sampled-token output supervision while using optimal transport to adaptively align heterogeneous teacher and student layers in a frozen shared low-rank space.

## Target and status

- Target style: ICLR/ICML-style machine-learning paper, approximately 9 pages before references.
- Current status: pre-results method paper; student, OPD, and OPRD baselines are complete. No final rank-8/32/64 EMD-OPD result artifact has been ingested and verified in this workspace; the versioned local fitter directly supports ranks 8 and 32, while rank 64 requires separate implementation provenance.
- Evidence boundary: no claim that EMD-OPD improves accuracy until its final evaluations finish.

## Claims-to-evidence matrix

| Claim | Evidence available now | Decisive missing evidence | Planned location |
|---|---|---|---|
| Representation supervision adds value when OPD is retained | OPD is much stronger than the student; OPRD is also stronger than the student but weaker than OPD | OPD+EMD must improve over OPD under matched settings | Introduction, Results |
| Adaptive many-to-many transport improves on fixed matching | A full 28-by-36 cost-dependent transport implementation and CPU gradient tests | OPD+fixed-map versus OPD+EMD with the identical bridge, rank, mask, normalization, coefficient, and budget | Method, Analysis |
| A shared low-rank bridge defines a common cross-width coordinate system with bounded cache size | Frozen shared PCA coordinate; analytical cache sizes and CPU tests | Held-out full-window diagnostics plus end-to-end memory and throughput | Method, Efficiency |
| EMD-OPD improves reasoning distillation | None yet | MATH500, AMC23, AIME24 final metrics, preferably multiple seeds | Abstract, Results |

## Narrative arc

1. **Problem.** OPD transfers next-token behavior but has no explicit intermediate-state matching loss. Representation-only distillation can exploit hidden structure, but it does not establish whether that signal adds value when OPD is retained or whether adaptive matching improves on a fixed map.
2. **Observation.** In the controlled Qwen3 setup, sampled-token OPD is the strongest completed baseline. Pure OPRD-Bridge improves over the student but trails OPD, so replacing OPD is not justified here.
3. **Idea.** Keep OPD unchanged. Map both models into one frozen low-rank coordinate system and use EMD to choose a many-to-many layer alignment for every minibatch.
4. **Test.** Compare EMD-OPD rank 8/32/64 against the completed student, OPD, and OPRD baselines under the same training prompts, data, and evaluation protocol.
5. **Decision rule.** Macro Avg@8 is the primary endpoint; all ranks and all three tasks are reported rather than selecting a rank on the final test. Any transport-specific claim additionally requires OPD+EMD to beat OPD+fixed-map under matched settings. Evaluation intervals will resample problems, not individual answers.

## Section outline

### Abstract

State the method and why fixed representation matching is insufficient. Report the completed baseline gap as motivation. Mark the current draft as pre-results and add one quantitative EMD-OPD sentence only after verified evaluation.

### 1. Introduction

- OPD gives dense supervision on student trajectories but operates at the output head.
- PKD and OPRD show that hidden representations contain useful supervision.
- Whether a cost-dependent many-to-many plan improves on fixed mapping is an open empirical question for a 28-layer student and 36-layer teacher.
- EMD-OPD retains the strongest local baseline and learns an adaptive transport plan.
- Contributions: method, shared bridge, controlled evaluation.

### 2. Related Work

Organize by: on-policy output distillation; intermediate representation distillation; optimal-transport layer matching. Clarify that EOPD is an entropy-aware output-space method and supplies the experimental protocol, whereas BERT-EMD supplies the many-to-many alignment idea.

### 3. Method

1. Student-generated trajectories and sampled-token OPD.
2. Frozen shared low-rank bridge.
3. Full student-teacher layer cost matrix over response tokens.
4. Exact unregularized balanced EMD with one detached plan per 32-trajectory optimizer minibatch; $\tau$ controls post-minibatch cost-attention marginal updates.
5. Combined loss and gradient path.
6. Complexity and bridge-rank tradeoff.

### 4. Experimental Setup

Document exact public EOPD-derived settings and every local adaptation. Include teacher/student identities, MATH train data, prompt, batch 128, minibatch 32, microbatch 1, AdamW 3e-6, cosine schedule, 3 epochs/174 rollout steps, response length 4096, train temperature 1 and top-p 1, and the frozen n=8 evaluation protocol.

### 5. Results

- Baseline table with student, sampled-token OPD, current OPRD-Bridge, and EMD-OPD ranks.
- Accuracy: Avg@8 and Pass@8 on MATH500, AMC23, AIME24.
- Efficiency: peak GPU memory, host memory, throughput, wall time.
- Do not fill rank-8/32/64 values until result files pass integrity checks.

### 6. Analysis

- Rank tradeoff (8/32/64).
- Learned transport matrix and transport entropy.
- Required matched set: OPD-only; shared-bridge fixed-map+OPD; shared-bridge EMD-only; shared-bridge EMD+OPD.
- Uniform/frozen-plan control to test whether cost-dependent transport itself matters.
- Response-length/truncation analysis.

### 7. Limitations

One training seed, fixed evaluation seed, three benchmarks, sampled-token OPD approximation, controlled protocol adaptation rather than strict EOPD or OPRD reproduction, and dependence on bridge calibration.

### 8. Conclusion

Summarize the design contribution. Make the performance conclusion conditional on the completed results.

## Figures and tables

1. **Figure 1:** student rollout passes through both sampled-token OPD and a shared-bridge layer transport path; detached teacher and transport plan are visually explicit.
2. **Table 1:** training and evaluation protocol.
3. **Table 2:** main benchmark results, with macro averages.
4. **Figure 2:** 28-by-36 learned transport matrix for the best rank.
5. **Figure 3:** accuracy versus peak memory for ranks 8/32/64.
6. **Table 3:** the required matched four-way comparison plus a uniform/frozen-plan control.

## Verified citations

- Jin et al. (2026), *Entropy-Aware On-Policy Distillation of Language Models*, arXiv:2603.07079.
- Yang et al. (2026), *OPRD: On-Policy Representation Distillation*, arXiv:2606.06021.
- Li et al. (2020), *BERT-EMD: Many-to-Many Layer Mapping for BERT Compression with Earth Mover's Distance*, EMNLP, DOI 10.18653/v1/2020.emnlp-main.242.
- Sun et al. (2019), *Patient Knowledge Distillation for BERT Model Compression*, EMNLP-IJCNLP, DOI 10.18653/v1/D19-1441.

## Writing constraints

- Do not describe current OPRD as literally OPD plus PKD; it is representation-only in the audited configuration.
- Do not call the work a strict EOPD reproduction.
- Do not compare local OPRD numbers directly with the OPRD paper as if protocols matched.
- Do not claim statistical significance from one training seed.
- Do not infer EMD-OPD effectiveness from bridge calibration cosine alone.
