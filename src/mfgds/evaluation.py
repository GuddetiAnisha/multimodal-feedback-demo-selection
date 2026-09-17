"""Task scores and query-clustered bootstrap summaries."""
import re
import numpy as np
import pandas as pd


def normalize_answer(text):
    text = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    words = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
             "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    return " ".join(words.get(w, w) for w in text.split() if w not in {"a", "an", "the"})


def score(prediction, example):
    if example.choices:
        text = prediction.strip()
        match = re.fullmatch(r"(?:[Aa]nswer\s*:\s*)?\(?([A-Z])\)?[.\s]*", text)
        if match:
            return float(match.group(1) == example.answer)
        # Permit exact option text without extracting a letter from free-form prose.
        idx = ord(example.answer)-65
        return float(normalize_answer(text) == normalize_answer(example.choices[idx]))
    p = normalize_answer(prediction)
    if example.answers:
        # Official leave-one-annotator-out consensus equation; normalization is simplified.
        answers = [normalize_answer(a) for a in example.answers]
        return float(np.mean([min(sum(p == a for j, a in enumerate(answers) if j != i)/3, 1)
                              for i in range(len(answers))]))
    return float(p == normalize_answer(example.answer))


GROUPS = ["variant", "method", "k", "ordering", "context_budget"]


def summarize(frame, bootstrap=1000):
    rows = []
    for keys, sub in frame.groupby(GROUPS, dropna=False):
        # Average seed repetitions first; resample query groups, not repeated rows.
        per_query = sub.groupby(["query_group", "query_id"])[["score", "gain"]].mean().reset_index()
        grouped = per_query.groupby("query_group")
        clusters = grouped[["score", "gain"]].sum().to_numpy()
        counts = grouped.size().to_numpy()
        rng = np.random.default_rng(12345)
        means = []
        for _ in range(bootstrap):
            sample = rng.integers(len(clusters), size=len(clusters))
            means.append(clusters[sample].sum(axis=0) / counts[sample].sum())
        means = np.asarray(means)
        lower, upper = np.quantile(means, [0.025, 0.975], axis=0)
        seed_means = sub.groupby("seed").score.mean()
        rows.append(dict(zip(GROUPS, keys)) | {
            "score_mean": float(sub.score.mean()), "score_ci_low": float(lower[0]), "score_ci_high": float(upper[0]),
            "gain_mean": float(sub.gain.mean()), "gain_ci_low": float(lower[1]), "gain_ci_high": float(upper[1]),
            "seed_std": float(seed_means.std(ddof=1)) if len(seed_means)>1 else 0.0,
            "seeds": len(seed_means), "query_groups": len(clusters), "observations": len(sub),
            **{name+"_mean": float(sub[name].mean()) for name in ["latency_s", "retrieval_s", "input_tokens", "output_tokens", "effective_k", "diversity", "redundancy"]}})
    return pd.DataFrame(rows)


def report(frame, output, mock):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = summarize(frame)
    summary.to_csv(output / "summary.csv", index=False)
    title = "MOCK / SYNTHETIC PIPELINE CHECK" if mock else "Measured experiment results"
    fields = [*GROUPS, "score_mean", "gain_mean", "score_ci_low", "score_ci_high"]
    table = summary[fields].round(4)
    lines = ["# " + title, "", "Confidence intervals: percentile bootstrap over query groups after averaging seeds.",
             "These intervals condition on the fixed data and trained models; seed variation is reported separately.", "",
             "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"]*len(fields)) + " |"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in table.itertuples(index=False, name=None)]
    (output / "table.md").write_text("\n".join(lines), encoding="utf-8")
    # Facets retain ordering/budget distinctions instead of pooling incompatible settings.
    for (variant, ordering, budget), sub in summary.groupby(["variant", "ordering", "context_budget"]):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method, group in sub.groupby("method"):
            group = group.sort_values("k")
            ax.plot(group.k, group.score_mean, marker="o", label=method)
            ax.fill_between(group.k, group.score_ci_low, group.score_ci_high, alpha=.1)
        ax.set(title=f"{title}\n{variant}, {ordering}, budget={budget}", xlabel="Requested demonstrations", ylabel="Task score", ylim=(-.05, 1.05))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"plot_{variant}_{ordering}_{budget}.png", dpi=140)
        plt.close(fig)
    return summary
