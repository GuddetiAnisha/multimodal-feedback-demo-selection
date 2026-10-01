"""Greedy maximum marginal relevance plus hard model-measured context limits."""
import numpy as np
import torch
from .embeddings import normalize
from .devices import resolve_device


def select(scores, vectors, k, redundancy=0.25, device="cpu"):
    if k < 0 or redundancy < 0:
        raise ValueError("k and redundancy must be nonnegative")
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Selection scores must be finite")
    device = resolve_device(device)
    if device.type == "cuda":
        with torch.inference_mode():
            values = torch.as_tensor(scores, dtype=torch.float32, device=device)
            v = torch.as_tensor(vectors, dtype=torch.float32, device=device)
            v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
            chosen = []
            remaining = torch.ones(len(values), dtype=torch.bool, device=device)
            penalty = torch.zeros_like(values)
            for _ in range(min(k, len(values))):
                objective = (values - redundancy * penalty).masked_fill(~remaining, -torch.inf)
                best = int(objective.argmax().item())
                chosen.append(best)
                remaining[best] = False
                penalty = torch.maximum(penalty, (v @ v[best]).clamp_min(0))
            return chosen
    vectors = normalize(vectors)
    chosen, remaining = [], list(range(len(scores)))
    while remaining and len(chosen) < k:
        def objective(i):
            penalty = max(0.0, float(np.max(vectors[chosen] @ vectors[i]))) if chosen else 0.0
            return float(scores[i]) - redundancy*penalty
        best = max(remaining, key=lambda i: (objective(i), -i))
        chosen.append(best)
        remaining.remove(best)
    return chosen


def order(indices, scores, mode, rng):
    indices = list(indices)
    if mode == "best_first":
        return sorted(indices, key=lambda i: (-scores[i], i))
    if mode == "best_last":
        return sorted(indices, key=lambda i: (scores[i], i))
    if mode == "random":
        rng.shuffle(indices)
        return indices
    raise ValueError(mode)


def fit_context(query, demos, model, budget):
    """Drop last demonstrations intact, preserving question and generation reserve."""
    demos = list(demos)
    while True:
        tokens = model.count_tokens(query, demos)
        if tokens + model.max_new_tokens <= budget:
            return demos, tokens
        if not demos:
            raise ValueError("Query plus generation reserve exceeds context budget")
        demos.pop()


def diversity(vectors, device="cpu"):
    if len(vectors) < 2:
        return 0.0, 0.0
    device = resolve_device(device)
    if device.type == "cuda":
        with torch.inference_mode():
            v = torch.as_tensor(vectors, dtype=torch.float32, device=device)
            v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
            i, j = torch.triu_indices(len(v), len(v), offset=1, device=device)
            sims = (v @ v.T)[i, j]
            return float((1-sims).mean().item()), float(sims.clamp_min(0).mean().item())
    v = normalize(vectors)
    sims = (v @ v.T)[np.triu_indices(len(v), k=1)]
    return float(np.mean(1-sims)), float(np.mean(np.maximum(sims, 0)))
