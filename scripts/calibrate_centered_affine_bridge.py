"""Fit the paper's frozen rank-8 centered-affine student bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA
from transformers import AutoModelForCausalLM


MAPPING = [round(i * 35 / 27) for i in range(28)]
RANK = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).cuda().eval()
    model.requires_grad_(False)
    return model


def selected_response_indices(pair_id: object, response_length: int, seed: int) -> torch.Tensor:
    if response_length <= 0:
        raise ValueError("calibration responses must contain at least one token")
    if response_length <= 64:
        return torch.arange(response_length)
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode()).digest()
    pair_seed = int.from_bytes(digest[:8], "little")
    indices = np.random.default_rng(pair_seed).choice(response_length, size=64, replace=False)
    return torch.from_numpy(np.sort(indices))


@torch.no_grad()
def hidden(model, pair: dict, layers: int, seed: int) -> torch.Tensor:
    prompt = pair["prompt_ids"]
    response = pair["response_ids"]
    ids = torch.tensor([prompt + response], device="cuda")
    output = model.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        output_hidden_states=True,
        use_cache=False,
    )
    positions = selected_response_indices(pair["id"], len(response), seed).to(ids.device)
    positions += len(prompt)
    return torch.stack([output.hidden_states[index + 1][0, positions] for index in range(layers)])


def split_pairs(path: Path, seed: int) -> tuple[list[dict], list[dict]]:
    pairs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(pairs) != 2000 or len({pair["id"] for pair in pairs}) != 2000:
        raise ValueError("Expected 2,000 unique pairs for the fixed 1,800/200 split")
    indices = np.arange(len(pairs))
    np.random.default_rng(seed).shuffle(indices)
    validation_indices = set(indices[:200].tolist())
    train = [pair for index, pair in enumerate(pairs) if index not in validation_indices]
    validation = [pair for index, pair in enumerate(pairs) if index in validation_indices]
    return train, validation


@torch.no_grad()
def cache_student(model, pairs: list[dict], seed: int) -> list[torch.Tensor]:
    result = []
    for index, pair in enumerate(pairs, start=1):
        result.append(hidden(model, pair, 28, seed).cpu().to(torch.bfloat16))
        if index % 20 == 0 or index == len(pairs):
            print("CACHE_STUDENT", json.dumps({"pairs": index, "total": len(pairs)}), flush=True)
    return result


@torch.no_grad()
def fit_teacher_basis(
    model, pairs: list[dict], seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pca = IncrementalPCA(n_components=RANK)
    for index, pair in enumerate(pairs, start=1):
        value = hidden(model, pair, 36, seed).reshape(-1, 4096).float().cpu().numpy()
        pca.partial_fit(value)
        if index % 20 == 0 or index == len(pairs):
            print("FIT_TEACHER_PCA", json.dumps({"pairs": index, "total": len(pairs)}), flush=True)
    return torch.from_numpy(pca.components_).float(), torch.from_numpy(pca.mean_).float()


@torch.no_grad()
def cache_teacher(
    model,
    pairs: list[dict],
    seed: int,
    teacher_projection: torch.Tensor,
    teacher_mean: torch.Tensor,
) -> list[torch.Tensor]:
    projection = teacher_projection.cuda()
    mean = teacher_mean.cuda()
    result = []
    for index, pair in enumerate(pairs, start=1):
        value = hidden(model, pair, 36, seed)
        result.append(F.linear(value.float() - mean, projection).cpu().to(torch.bfloat16))
        if index % 20 == 0 or index == len(pairs):
            print("CACHE_TEACHER", json.dumps({"pairs": index, "total": len(pairs)}), flush=True)
    return result


def layer_mean(cache: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros(cache[0].shape[0], cache[0].shape[2], dtype=torch.float64)
    count = 0
    for value in cache:
        total += value.double().sum(1)
        count += value.shape[1]
    return (total / count).float()


def fit_centered_affine(
    student_cache: list[torch.Tensor],
    teacher_cache: list[torch.Tensor],
    student_mean: torch.Tensor,
    target_mean: torch.Tensor,
) -> torch.Tensor:
    """Solve each layer's centered least-squares problem via normal equations."""
    weights = torch.empty(28, RANK, 2048)
    for student_layer, teacher_layer in enumerate(MAPPING):
        gram = torch.zeros(2048, 2048, dtype=torch.float64)
        cross = torch.zeros(2048, RANK, dtype=torch.float64)
        for student, teacher in zip(student_cache, teacher_cache):
            source = student[student_layer].double() - student_mean[student_layer].double()
            target = teacher[teacher_layer].double() - target_mean[student_layer].double()
            gram.addmm_(source.T, source)
            cross.addmm_(source.T, target)
        solution = torch.linalg.lstsq(gram, cross).solution
        if not torch.isfinite(solution).all():
            raise FloatingPointError(f"non-finite least-squares solution at student layer {student_layer}")
        weights[student_layer] = solution.T.float()
        print("FIT_LAYER", json.dumps({"student_layer": student_layer}), flush=True)
    return weights


@torch.no_grad()
def validation_metrics(
    weights: torch.Tensor,
    student_cache: list[torch.Tensor],
    teacher_cache: list[torch.Tensor],
    student_mean: torch.Tensor,
    target_mean: torch.Tensor,
) -> dict[str, float]:
    squared = cosine = 0.0
    positions = 0
    mapping = torch.tensor(MAPPING)
    for student, teacher in zip(student_cache, teacher_cache):
        projected = torch.einsum(
            "lnd,lrd->lnr", student.float() - student_mean[:, None, :], weights
        ) + target_mean[:, None, :]
        target = teacher[mapping].float()
        squared += (F.normalize(projected, dim=-1) - F.normalize(target, dim=-1)).square().sum().item()
        cosine += F.cosine_similarity(projected, target, dim=-1).sum().item()
        positions += projected.shape[0] * projected.shape[1]
    return {
        "mapped_normalized_mse": squared / (positions * RANK),
        "mapped_cosine": cosine / positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one admitted GPU with CUDA_VISIBLE_DEVICES")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_pairs, validation_pairs = split_pairs(args.pairs, args.seed)
    all_pairs = train_pairs + validation_pairs

    teacher_model = load_model(args.teacher_model)
    teacher_projection, teacher_mean = fit_teacher_basis(
        teacher_model, train_pairs, args.seed
    )
    teacher_all = cache_teacher(
        teacher_model, all_pairs, args.seed, teacher_projection, teacher_mean
    )
    del teacher_model
    torch.cuda.empty_cache()

    student_model = load_model(args.student_model)
    student_all = cache_student(student_model, all_pairs, args.seed)
    del student_model
    torch.cuda.empty_cache()

    student_train, student_validation = student_all[:1800], student_all[1800:]
    teacher_train, teacher_validation = teacher_all[:1800], teacher_all[1800:]
    student_mean = layer_mean(student_train)
    teacher_layer_mean = layer_mean(teacher_train)
    target_mean = teacher_layer_mean[torch.tensor(MAPPING)]
    weights = fit_centered_affine(
        student_train, teacher_train, student_mean, target_mean
    )
    metrics = validation_metrics(
        weights, student_validation, teacher_validation, student_mean, target_mean
    )

    artifact = {
        "complete": True,
        "kind": "shared_teacher_pca_centered_affine_emd_v1",
        "rank": RANK,
        "mapping": MAPPING,
        "pt": teacher_projection,
        "mean": teacher_mean,
        "ps": weights,
        "student_mean": student_mean,
        "target_mean": target_mean,
        "calibration_fit_examples": 1800,
        "calibration_validation_examples": 200,
        "calibration_positions_per_example": 64,
        "provenance": {
            "pairs_sha256": sha256(args.pairs),
            "seed": args.seed,
            "teacher_pca": "shared rank-8 IncrementalPCA over all layers and sampled fitting positions",
            "student_fit": "centered least squares",
            "affine_formula": "(h_s - mu_s[layer]) @ W_s[layer]^T + nu_t[mapping[layer]]",
            "validation": metrics,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(sha256(args.output))


if __name__ == "__main__":
    main()
