"""Greedy maximum marginal relevance plus hard model-measured context limits."""
import numpy as np
from .embeddings import normalize


def select(scores, vectors, k, redundancy=0.25):
    if k < 0 or redundancy < 0:
        raise ValueError("k and redundancy must be nonnegative")
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError("Selection scores must be finite")
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


def diversity(vectors):
    if len(vectors) < 2:
        return 0.0, 0.0
    v = normalize(vectors)
    sims = (v @ v.T)[np.triu_indices(len(v), k=1)]
    return float(np.mean(1-sims)), float(np.mean(np.maximum(sims, 0)))
