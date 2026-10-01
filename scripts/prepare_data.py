"""Convert official, locally downloaded benchmarks to canonical JSONL splits."""
import argparse
import json
from pathlib import Path
from mfgds.data import scienceqa, vqav2, partition, save_jsonl, validate_splits, leakage_safe_training_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["scienceqa", "vqav2"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Per-split cap after partitioning; 0 retains all")
    args = parser.parse_args()
    if args.benchmark == "scienceqa":
        # Hold out official validation for final evaluation; partition train by group.
        official_train, official_eval = scienceqa(args.root, "train"), scienceqa(args.root, "val")
    else:
        def load(split):
            return vqav2(args.root / f"v2_OpenEnded_mscoco_{split}_questions.json",
                         args.root / f"v2_mscoco_{split}_annotations.json", args.root / split, split)
        official_train, official_eval = load("train2014"), load("val2014")
    training, evaluation, audit = leakage_safe_training_pool(official_train, official_eval)
    parts = partition(training, args.seed, (.8, .2))
    splits = [parts[0], parts[1], evaluation]
    if args.limit:
        if args.limit < 1:
            raise ValueError("Limit must be positive")
        splits = [rows[:args.limit] for rows in splits]
    validate_splits(*splits)
    for name, rows in zip(["demos", "feedback", "evaluation"], splits):
        save_jsonl(rows, args.output / (name + ".jsonl"))
    audit.update(benchmark=args.benchmark, seed=args.seed, limit=args.limit,
                 prepared_rows={name: len(rows) for name, rows in zip(["demos", "feedback", "evaluation"], splits)})
    (args.output / "preparation_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(audit["prepared_rows"])
    print(f"Excluded {audit['excluded_training_rows']} training rows overlapping held-out evaluation groups.")
    print("No sample cap applied." if not args.limit else f"Applied per-split sample cap: {args.limit}.")
    print(f"Preparation audit: {args.output / 'preparation_report.json'}")


if __name__ == "__main__":
    main()
