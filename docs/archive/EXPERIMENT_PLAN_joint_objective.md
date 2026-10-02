# Experiment Plan

**Problem**: 在锁定的 EOPD 公共训练协议下，判断在 sampled-token OPD 上加入低秩共享空间中的 layer-level EMD，能否比纯 OPD 和固定层映射表示蒸馏更稳定地提升数学推理。

**Method Thesis**: `EMD-OPD = sampled-token OPD + shared-bridge layer EMD`。真正需要验证的不是“多加一个 loss 是否有效”，而是“在相同表示空间、token mask、训练预算和损失尺度下，数据依赖的 many-to-many layer transport 是否优于 fixed layer matching”。

**Date**: 2026-09-30

## 当前证据快照

- Student、sampled-token OPD、修复后 OPRD-Bridge 已完成统一协议评测。三项 Avg@8 分别为：Student `4.150/2.8125/0.4167`，OPD `62.175/34.0625/7.0833`，OPRD-Bridge `23.525/16.5625/2.500`。
- Teacher 统一协议评测已完成：MATH500 `84.125/95.2`、AMC23 `67.5/92.5`、AIME24 `25.833/53.333`，每格为 Avg@8/Pass@8（百分比）。
- EMD-OPD rank=32 已完成但低于 OPD；rank=8 与 rank=64 正在 175 服务器训练，尚不能冻结主结论。
- GPU7 正在准备的 `OPRD final_hf -> 174-step OPD` 使用一个已经完成 174 步的 OPRD 模型，再训练 174 步 OPD，因此总训练预算是 **348 步**。它是论文式顺序训练/恢复能力实验，不是 174 步公平主表行。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：EMD-OPD 在相同 174 步预算下优于 OPD | 决定方法是否有最终性能贡献 | 最终方法在三种子 macro Avg@8 上超过 OPD，且至少两个数据集不退化 | B1, B4 |
| C2：收益来自 adaptive layer transport | 隔离论文的机制创新 | 在同一 shared bridge、rank、mask、normalization、lambda 下，EMD+OPD 超过 fixed-map+OPD | B2 |
| C3：低秩两遍式实现具有可接受成本 | 支撑方法的实用性 | 报告校准成本、训练吞吐、GPU/主存峰值，并与 OPD、fixed-map+OPD 对齐 | B1, B4 |
| Anti-claim：收益来自额外训练步数、不同 bridge、不同 token mask 或更大梯度 | 审稿人最直接的替代解释 | 174 步 matched control；单独报告 348 步顺序实验；记录梯度范数与余弦 | B2, B3 |

## 冻结的公共协议

- Teacher 和 Student 身份、训练数据、提示词、rollout 与评分器均来自现有 EOPD 锁定配置。
- batch=128，mini-batch=32，micro-batch=1，3 epochs / 174 rollout steps，AdamW `3e-6` + cosine，response length=4096，seed=42。
- 评测固定为 MATH500、AMC23、AIME24；n=8，temperature=1，top_p=0.8，max_tokens=8192；主指标为 Avg@8，辅助指标为 Pass@8、长度、截断率和重复度。
- 机制对照固定 shared bridge rank=8，并使用相同 response-token mask、cost normalization、lambda 和训练步数。历史 OPRD 的独立 layer-pair bridge 与 last-2000 mask 只能作为方法基线，不能替代 matched mechanism control。
- rank=8/32/64 在固定 lambda=1 下同时改变表示容量、MSE 尺度和有效梯度强度，只能称配置消融，不能直接声称“rank 本身”的因果效果。

## 主表与机制表方法矩阵

| ID | 方法 | 预算 | 表中作用 |
|---|---|---:|---|
| M0 | Student / Teacher | 0 | 下界与上界 |
| M1 | sampled-token OPD | 174 | 核心强基线 |
| M2 | OPRD-Bridge-only | 174 | 历史表示-only基线 |
| M3 | OPRD作者式 Bridge + OPD top-1 | 174 | 强方法基线；回答“OPRD低是否因缺少OPD” |
| M4 | shared bridge fixed-map + OPD | 174 | **最关键机制对照**；与主方法只差 layer transport |
| M5 | shared bridge EMD-only | 174 | 分解 output loss 与 EMD loss 的贡献 |
| M6 | shared bridge EMD+OPD, rank=8 | 174 | 主方法 |
| M7 | shared bridge EMD+OPD, rank=32/64 | 各174 | 配置/容量消融 |
| M8 | 87步 OPRD -> 87步 OPD | 174 | 同总预算的 staged control |
| M9 | 174步 OPRD -> 174步 OPD | 348 | 当前 GPU7 顺序实验；只进 appendix/额外预算表 |
| M10 | OPD-only 348步 | 348 | 若 M9 用于性能论证，则必须增加的预算对照 |

M3 与 M4 不能合并。M3 检查 OPRD 论文式联合目标是否是强基线；M4 使用与 M6 完全相同的 shared bridge 和数据处理，只把 EMD transport 换为 fixed proportional matching，才真正隔离 C2。

## Experiment Blocks

### B0：完成并审计当前运行

- **Runs**：rank=8、rank=64、rank=32、Teacher、GPU7 的 348 步顺序实验。
- **Required checks**：checkpoint identity、配置 hash、完整 500/40/30 题与 8 rollouts、无重复样本、Avg@8/Pass@8 重算、训练/评测 prompt 逐字段一致。
- **Decision gate**：只有全部审计完成后，才决定主方法 rank 与是否需要 lambda 调整。
- **Priority**：MUST-RUN。

### B1：主结果与效率

- **Claim tested**：至少一个 EMD-OPD 配置能在 174 步预算下超过纯 OPD。
- **Compared systems**：M0、M1、M2、M3、M6；M7 放消融表。
- **Metrics**：三项 Avg@8/Pass@8、macro Avg@8、响应长度、截断率、wall time、tokens/s、GPU/主存峰值、bridge calibration time。
- **Success criterion**：M6 在 macro Avg@8 上超过 M1，并至少两个数据集不退化；最终结论须由三种子确认。
- **Failure interpretation**：若 rank=8/32/64 均低于 OPD，则当前 `lambda=1` joint objective 没有形成性能贡献；先做 B3，不继续盲目扩大 rank。
- **Table target**：主表 + 效率表。
- **Priority**：MUST-RUN。

### B2：机制归因的三个必要对照

1. **M3：OPRD Bridge + OPD top-1**。复现论文方向的强基线，但保留其原始 bridge/mask 语义。
2. **M4：shared bridge fixed-map + OPD**。与 M6 共用 rank=8 artifact、all-response mask、normalization、lambda 和174步预算，只把 `F` 换为 proportional one-hot/fixed flow。
3. **M5：shared bridge EMD-only**。关闭 OPD，检验 EMD 是否独立有用以及联合收益是否只是 OPD 主导。

- **Optional routing control**：uniform/frozen transport plan。只有 M6 超过 M4 时再运行，用于证明收益来自 cost-dependent routing，而非 dense averaging。
- **Success criterion**：`M6 > M4` 支撑 adaptive matching；`M6 > M5` 支撑保留 output supervision；`M3 > M2` 说明历史 OPRD 的低分部分来自缺少 OPD。
- **Failure interpretation**：若 M4≈M6，论文贡献应改写为 shared representation regularization，而不能主张 EMD routing 有效。
- **Priority**：M3、M4 MUST-RUN；M5 SHOULD-RUN；routing control CONDITIONAL。

### B3：梯度冲突与 lambda 决策

- **先诊断，不先扫参**：在相同 checkpoint 和相同4个 mini-batches 上分别反传 OPD 与 EMD，记录 `||g_OPD||`、`||g_EMD||`、cosine、有效比例 `lambda||g_EMD||/||g_OPD||`，分 early/mid/late checkpoint 报告。
- **Decision rule**：若 lambda=1 的 EMD 梯度显著大于 OPD 或长期负余弦，再从 `{0.1, 0.3}` 中按诊断选择一个完整174步 run；不无条件同时跑两个。
- **可以探索的改进**：若 joint loss 冲突明显，优先研究 warm-up/annealed lambda 或 conflict-aware weighting。它们属于新方法版本，必须与固定 lambda=1 分开命名。
- **Success criterion**：用梯度证据解释 rank/性能趋势，并预注册下一次完整 run 的 lambda。
- **Figure target**：loss、gradient-ratio、gradient-cosine 随训练步数曲线。
- **Priority**：若当前 ranks 未超过 OPD，则 MUST-RUN。

### B4：预算、公平性与稳健性

- **顺序训练**：当前 M9 总预算348步，只能与 M10（348步 OPD）直接比较。若要进入174步主表，另跑 M8（87+87）。第二阶段从合并 HF 权重初始化，并重新初始化 optimizer/scheduler。
- **随机种子**：筛选阶段只用 seed42；最终保留 M1、M4、最佳 M6 和必要时 M3，各补 seed43/44。报告均值、标准差和 problem-level bootstrap 95% CI。
- **真实 EOPD baseline**：当前实现是 sampled-token OPD，并未包含 EOPD entropy-gated forward-KL。若论文要声称优于 EOPD，必须实现并运行真实 EOPD；否则全文只能说“使用 EOPD 的公共设置”，不能把 M1 标成 EOPD。
- **第二模型对**：只有当主结果为正且算力允许时，再增加一个 teacher/student pair；这是增强外部有效性的 NICE-TO-HAVE，不阻塞第一版主结论。
- **Priority**：预算对齐和三种子 MUST-RUN；真实 EOPD baseline 取决于最终 claim；第二模型对 NICE-TO-HAVE。

## Run Order and Decision Gates

| 顺序 | Run | 目的 | Go / Stop Gate | 预计单卡时间 |
|---:|---|---|---|---:|
| 0 | 完成 rank=8/64、审计 rank=32 与 Teacher | 冻结当前证据 | 三项结果均完整可复算 | 已在运行/已完成 |
| 1 | M3：OPRD Bridge + OPD top-1 | 补齐论文强基线 | 判断 representation+output 是否优于 OPD | 约22–26h |
| 2 | M4：shared fixed-map + OPD | 隔离 EMD routing | 与 M6 只差 transport | 约22–26h |
| 3 | 梯度诊断 | 解释负结果并选择 lambda | 无冲突则不扫 lambda | 数小时以内，可复用 checkpoint |
| 4 | M5 或一个诊断选定的 lambda run | 完成加法分解/修复尺度 | 只跑能改变论文结论的一项 | 约22–26h |
| 5 | M8：87->87 或 M10：OPD 348 | 对齐顺序训练预算 | 取决于是否使用 M9 作主张 | 约22–52h |
| 6 | 最终方法补 seed43/44 | 形成论文级统计证据 | 结论跨种子一致 | 每方法约44–52h |
| 7 | 真实 EOPD / 第二模型对 | 增强强基线与外部有效性 | 仅在目标 claim 需要时运行 | 每项约22–26h以上 |

## 结果驱动的论文路线

1. **M6 > M1 且 M6 > M4**：保留“adaptive cross-layer transport improves OPD”为主结论，进入三种子、transport heatmap 和效率分析。
2. **M6 > M4 但 M6 < M1**：只能主张 EMD 优于 fixed representation matching；论文需转向梯度冲突或 staged/annealed objective，不能宣称提升 OPD。
3. **M4≈M6 且二者优于 M1**：贡献是 shared representation regularization，不是 EMD；标题和方法主张必须改。
4. **所有 joint 方法低于 M1，但 M9 恢复明显**：主线转为“representation pretraining + output realignment”的 staged distillation，并用 M8/M10 做预算对照。
5. **所有方法均低于 M1，且顺序训练也无恢复**：停止扩大实验矩阵；将结果作为失败诊断，不继续为原假设补实验。

## 论文还需要完成的内容

- **主图**：student rollout -> OPD token loss 与 shared bridge -> layer cost -> detached OT plan 两条支路，清楚标出 teacher/no-grad、frozen bridge 和 student gradient path。
- **主表**：Teacher、Student、OPD、OPRD-only、OPRD+OPD、shared-fixed+OPD、EMD-OPD；348步顺序结果单列，不与174步表混排。
- **机制图**：transport heatmap、layer marginals/entropy、OPD/EMD gradient cosine；展示是否真的出现 many-to-many routing。
- **效率表**：参数量、额外可训练参数、校准耗时、训练时长、GPU峰值、主存峰值、吞吐。
- **Related Work**：至少覆盖 on-policy output KD（OPD/EOPD）、representation KD（PKD、TinyBERT/MiniLM、OPRD）、optimal-transport KD（BERT-EMD及后续相关工作）。提交前做正式 novelty search，确认“on-policy + shared low-rank bridge + adaptive layer OT”没有被近期工作覆盖。
- **Claim discipline**：当前只能写“使用 EOPD 公共配置”；除非补真实 EOPD run，不能写“优于 EOPD”。
- **统计与完整性**：逐样本结果、三种子、bootstrap CI、配置/代码/artifact hash、失败运行记录、选择 rank/lambda 的规则。
- **局限性**：单模型对、数学领域、bridge calibration prior、rank与loss scale耦合、额外计算和小数据集评测方差。

## Compute and Data Budget

- **最小可发表闭环（当前 runs 之外）**：M3 + M4 + 一个 B3 决定的完整 run + 最终入选方法的额外种子。
- **典型完整预算**：3个必要新单种子 run 约66–78 GPU-hours；若4个方法各补2个种子，另需约176–208 GPU-hours。
- **磁盘**：每个活跃实验保留一个最近完整 checkpoint 与最终 HF；评测 JSONL 和配置/hash 永久保留。
- **最大瓶颈**：不是显存，而是每个174步 run 约一天以及三种子总成本。

## Final Checklist

- [x] Teacher统一协议三项评测完成
- [ ] rank=8、rank=32、rank=64全部完成结果审计
- [ ] OPRD Bridge + OPD top-1 强基线完成
- [ ] shared bridge fixed-map + OPD matched control 完成
- [ ] EMD-only或等价加法分解完成
- [ ] 348步顺序结果与174步/348步预算对照正确分表
- [ ] 梯度比例和冲突诊断完成
- [ ] 最终方法完成3个训练种子
- [ ] 主表、机制图、效率表完成
- [ ] 真实 EOPD claim 与实际 baseline 保持一致
- [ ] novelty search、citation audit、result-to-claim audit 完成
