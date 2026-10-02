from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "verl"))

ANSWER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
BREVITY_INSTRUCTION = (
    "Keep your reasoning concise, avoid repetition, and complete your response "
    "within {max_tokens} tokens."
)


def add_answer_instruction(messages: list[dict], max_tokens: int = 8192) -> list[dict]:
    updated = [dict(message) for message in messages]
    brevity = BREVITY_INSTRUCTION.format(max_tokens=max_tokens)
    for message in reversed(updated):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            content = message["content"].rstrip()
            if not content.endswith(brevity):
                if not content.endswith(ANSWER_INSTRUCTION):
                    content = f"{content}\n\n{ANSWER_INSTRUCTION}"
                content = f"{content} {brevity}"
            message["content"] = content
            return updated
    raise ValueError("chat messages lack a text user message")


def load_rows(task: str, path: str, max_tokens: int = 8192) -> list[dict]:
    source = Path(path)
    rows = (
        [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if source.suffix == ".jsonl"
        else pd.read_parquet(source).to_dict("records")
    )
    parsed = []
    for index, row in enumerate(rows):
        messages = row.get("prompt")
        reward_model = row.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth")
        if messages is None or isinstance(messages, (str, bytes)) or ground_truth is None:
            raise ValueError(f"{task} row {index} lacks structured prompt/reward_model fields")
        messages = list(messages)
        if not messages or not all(isinstance(message, dict) for message in messages):
            raise ValueError(f"{task} row {index} has invalid chat messages")
        parsed.append(
            {
                "question_id": str(row.get("id", row.get("extra_info", {}).get("index", index))),
                "messages": add_answer_instruction(messages, max_tokens=max_tokens),
                "ground_truth": str(ground_truth),
            }
        )
    return parsed


def summarize(path: Path, task: str, n: int = 8) -> dict:
    by_question: dict[str, list[bool]] = {}
    seen = set()
    response_tokens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["task"] != task:
            continue
        identity = (row["question_id"], row["rollout_id"])
        if identity in seen:
            raise ValueError(f"duplicate rollout: {identity}")
        seen.add(identity)
        by_question.setdefault(row["question_id"], []).append(bool(row["correct"]))
        response_tokens.append(int(row["response_tokens"]))
    if not by_question or any(len(scores) != n for scores in by_question.values()):
        raise ValueError(f"expected exactly {n} rollouts per question")
    avg8 = sum(sum(scores) / len(scores) for scores in by_question.values()) / len(by_question)
    pass8 = sum(any(scores) for scores in by_question.values()) / len(by_question)
    return {
        "task": task,
        "questions": len(by_question),
        "rollouts": sum(map(len, by_question.values())),
        f"avg_at_{n}": avg8,
        f"pass_at_{n}": pass8,
        "mean_response_tokens": sum(response_tokens) / len(response_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score.ttrl_math import compute_score

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = load_rows(args.task, args.data, max_tokens=args.max_tokens)
    rows = rows[: args.limit] if args.limit else rows
    prompts = [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for row in rows
    ]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=10240,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        max_num_seqs=32,
        seed=args.seed,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(
            n=args.n,
            temperature=1.0,
            top_p=0.8,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
        use_tqdm=True,
    )
    if len(outputs) != len(rows) or any(len(result.outputs) != args.n for result in outputs):
        raise ValueError("generation returned incomplete questions or rollouts")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row, prompt, result in zip(rows, prompts, outputs):
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            for rollout_id, output in enumerate(result.outputs):
                grade = compute_score(output.text, row["ground_truth"], fast=False)
                record = {
                    "task": args.task,
                    "question_id": row["question_id"],
                    "rollout_id": rollout_id,
                    "gold_answer": row["ground_truth"],
                    "model_output": output.text,
                    "prediction": grade["pred"],
                    "correct": bool(grade["acc"]),
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": len(output.token_ids),
                    "finish_reason": getattr(output, "finish_reason", None),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = summarize(output_path, args.task, n=args.n)
    metrics["protocol"] = {
        "model": args.model,
        "data": args.data,
        "seed": args.seed,
        "n": args.n,
        "temperature": 1.0,
        "top_p": 0.8,
        "max_tokens": args.max_tokens,
        "engine": {"enforce_eager": True, "max_num_seqs": 32, "gpu_memory_utilization": 0.70},
        "answer_instruction": f"{ANSWER_INSTRUCTION} {BREVITY_INSTRUCTION.format(max_tokens=args.max_tokens)}",
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
