# Uniform-Marginal OT → OPD

## Task

Train Qwen3-1.7B-Base with a frozen Qwen3-8B teacher in three stages: fit a frozen rank-8 centered-affine representation bridge, perform cost-dependent representation alignment with an unregularized balanced optimal-transport plan and fixed uniform layer marginals, then start a separate sampled-token on-policy distillation stage from the aligned checkpoint with a fresh AdamW optimizer and cosine schedule.

## Environment installation

```bash
git clone https://github.com/ShenzhiYang2000/OPRD.git third_party/OPRD
git -C third_party/OPRD checkout <OPRD_COMMIT>
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -r src/verl/. third_party/OPRD/verl/verl/
python -m pip install -e third_party/OPRD/verl
```

## Training and testing commands

Replace every `<SET_...>` value in the configuration files, set `lambda_emd`, `bridge_sha256`, and every `solver` value in `config/method.json`, and then run. Bridge calibration requires exactly one visible GPU. The Stage-II implementation requires one GPU, one node, and static microbatches; these are execution constraints, not claims from the paper.

```bash
python scripts/calibrate_centered_affine_bridge.py \
  --pairs <CALIBRATION_PAIRS_JSONL> \
  --student-model <QWEN3_1_7B_BASE> \
  --teacher-model <QWEN3_8B> \
  --output artifacts/bridge.pt \
  --seed <BRIDGE_CALIBRATION_SEED>

python scripts/train.py --config config/emd_train.json
python scripts/train.py --config config/opd_train.json

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model <MODEL_OR_CHECKPOINT> \
  --task <TASK_NAME> \
  --data <TEST_PARQUET_OR_JSONL> \
  --output outputs/predictions.jsonl \
  --summary outputs/summary.json \
  --n 8 \
  --seed <EVALUATION_SEED> \
  --thinking-mode <enabled-or-disabled> \
  --max-model-len <TOTAL_CONTEXT_LENGTH> \
  --gpu-memory-utilization <GPU_MEMORY_FRACTION> \
  --max-num-seqs <MAX_CONCURRENT_SEQUENCES> \
  --engine-mode <eager-or-graph>

python -m pytest -q
```

## Dataset description

Training uses 7,500 problems sampled from the MATH training split in the VERL/OPRD Parquet format. Bridge fitting uses 1,800 student-generated prompt–response pairs, with a separate 200-pair held-out validation split; each JSONL record contains unique `id`, `prompt_ids`, and `response_ids` fields, and at most 64 response positions are sampled per example. Evaluation expects MATH-500, AMC 2023, Minerva, OlympiadBench, AIME 2024, and AIME 2025 records with structured `prompt` and `reward_model.ground_truth` fields. Datasets are not included.
