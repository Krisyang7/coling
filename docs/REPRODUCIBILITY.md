# Reproducibility record

## Method schedule

Each rank follows the same sequential schedule:

1. Load the student initialization and a frozen shared-coordinate Bridge.
2. Train for 174 steps with `lambda_emd * hidden_emd_loss` only.
3. Merge the final EMD checkpoint into a Hugging Face model.
4. Start a fresh AdamW optimizer and cosine schedule from that model.
5. Train for 174 steps with the original sampled-token OPD loss only.
6. Merge and evaluate MATH500, AMC23, and AIME24.

The transport plan is solved exactly as a balanced linear program, detached, and used in a differentiable recomputation of the student-side layer cost. Teacher representations are detached. The frozen Bridge has no optimizer state or gradients.

## Training protocol

| Setting | Value |
|---|---|
| Teacher | `Qwen/Qwen3-8B` |
| Student | `Qwen/Qwen3-1.7B-Base` |
| Training examples | 7,500 |
| Global batch | 128 |
| PPO mini-batch | 32 |
| Micro-batch per GPU | 1 |
| Epochs per stage | 3 |
| Optimizer steps per stage | 174 |
| Learning rate | `3e-6` |
| LR schedule | cosine, no warmup |
| Weight decay | `0.01` |
| Max prompt length | 2048 |
| Max training response length | 4096 |
| Rollout temperature / top-p | `1.0 / 1.0` |
| Seed | 42 |
| `lambda_emd` / `tau` | `1.0 / 1.0` |
| EMD layers / positions | all model block outputs / all valid response positions |

The same prepared training parquet was used for every comparison method, so the stored prompt field is byte-identical across runs. The dataset itself is not redistributed in this repository.

## Evaluation protocol

Each question receives eight sampled responses using temperature `1.0`, `top_p=0.8`, seed 42, and a maximum of 8192 generated tokens. The exact appended instruction is:

```text
Please reason step by step, and put your final answer within \boxed{}. Keep your reasoning concise, avoid repetition, and complete your response within 8192 tokens.
```

The evaluator and prompt are unchanged across the teacher, student, OPD, OPRD, and Bridge-EMD comparisons.

## Validated environment witness

The active runs used the versions recorded in [`requirements-validated.txt`](../requirements-validated.txt). This file records the observed environment; it is not a claim that every CUDA system can be reproduced by installing those packages without platform-specific constraints.

The implementation files in [`code/verl_extensions/`](../code/verl_extensions/) are copied from the exact run tree. They are intended to overlay the corresponding modules in the OPRD VERL source tree. The reference rank-8 driver under [`experiments/rank8/`](../experiments/rank8/) shows the actual stage order, validation witnesses, checkpoint policy, merge, and evaluation calls.

## Frozen Bridge integrity

The three Bridge tensors are stored under `experiments/rank*/artifacts/bridge.pt`. Verify them with the SHA-256 values in [`experiments/bridge_sha256.json`](../experiments/bridge_sha256.json) before launching a run.

These Bridges use one shared teacher PCA coordinate system and separately fitted student projections. Their ranks are the only intended difference in the sweep.
