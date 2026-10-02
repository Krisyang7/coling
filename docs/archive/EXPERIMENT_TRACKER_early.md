# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| R010 | Baseline | Teacher上界 | EOPD锁定Teacher | Avg@8, Pass@8, length | MUST | DONE | MATH 84.125/95.2；AMC 67.5/92.5；AIME 25.833/53.333 |
| R000 | Main | 完成主方法 | EMD-OPD r=8 | Avg@8, Pass@8, cost | MUST | RUNNING | 175 GPU6；2026-09-30 12:16为116/174 |
| R001 | Main | rank配置消融 | EMD-OPD r=32 | Avg@8, Pass@8, cost | MUST | DONE-PENDING-AUDIT | 已知低于OPD；需补结果完整性审计 |
| R002 | Main | rank配置消融 | EMD-OPD r=64 | Avg@8, Pass@8, cost | MUST | RUNNING | 175 GPU4；2026-09-30 12:16为84/174 |
| R201 | Baseline | 论文强基线 | OPRD Bridge + sampled-token OPD top-1 | Avg@8, Pass@8 | MUST | TODO | 保留OPRD原始bridge/mask语义，174步 |
| R203 | Mechanism | 隔离adaptive transport | shared bridge fixed-map + OPD, r=8 | Avg@8, gradients | MUST | TODO | 与r=8主方法只差layer transport |
| R204 | Mechanism | 分解联合目标 | shared bridge EMD-only, r=8 | Avg@8, gradients | SHOULD | TODO | 判断OPD保留是否必要 |
| R205 | Mechanism | 检查cost-dependent routing | uniform或frozen plan + OPD | Avg@8, transport | CONDITIONAL | BLOCKED | 仅当EMD超过fixed-map后运行 |
| R301 | Diagnose | 测梯度尺度与冲突 | OPD vs EMD gradients on matched batches | norm, cosine, ratio | MUST-IF-NEGATIVE | TODO | early/mid/late checkpoint；先诊断后选lambda |
| R302 | Tune | 修复loss尺度 | EMD-OPD r=8, lambda=0.1或0.3 | Avg@8, gradients | CONDITIONAL | BLOCKED | 只跑R301选择的一项 |
| R202 | Budget | 174步顺序对照 | 87步OPRD -> 87步OPD | Avg@8, Pass@8 | MUST-IF-STAGED-CLAIM | TODO | 新optimizer/scheduler，总预算174 |
| R501 | Appendix | 论文式完整顺序训练 | 174步OPRD -> 174步OPD | Avg@8, Pass@8 | APPENDIX | RUNNING-PREPARING | 175 GPU7；总预算348，不能直接与OPD174比较 |
| R502 | Budget | 348步预算对照 | OPD-only 348步 | Avg@8, Pass@8 | MUST-IF-R501-CLAIMED | TODO | 与R501直接公平比较 |
| R401 | Robustness | 最终三种子 | OPD / shared-fixed+OPD / best EMD /必要时OPRD+OPD | mean, std, CI | MUST | BLOCKED | seed42筛选后补43/44 |
| R402 | Strong baseline | 验证真实EOPD | entropy-gated EOPD | Avg@8, Pass@8 | CLAIM-DEPENDENT | TODO | 若声称优于EOPD则必须跑；否则仅称使用EOPD设置 |
| R403 | Generalization | 第二模型对 | best method on another teacher/student pair | Avg@8, cost | NICE | BLOCKED | 仅主结果为正且预算允许时运行 |
