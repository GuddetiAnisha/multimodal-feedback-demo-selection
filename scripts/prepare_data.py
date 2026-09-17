"""Convert official, locally downloaded benchmarks to canonical JSONL splits."""
import argparse
from pathlib import Path
from mfgds.data import scienceqa, vqav2, partition, save_jsonl, validate_splits


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
        parts = partition(scienceqa(args.root, "train"), args.seed, (.8, .2))
        splits = [parts[0], parts[1], scienceqa(args.root, "val")]
    else:
        def load(split):
            return vqav2(args.root / f"v2_OpenEnded_mscoco_{split}_questions.json",
                         args.root / f"v2_mscoco_{split}_annotations.json", args.root / split, split)
        parts = partition(load("train2014"), args.seed, (.8, .2))
        splits = [parts[0], parts[1], load("val2014")]
    if args.limit:
        if args.limit < 1:
            raise ValueError("Limit must be positive")
        splits = [rows[:args.limit] for rows in splits]
    validate_splits(*splits)
    for name, rows in zip(["demos", "feedback", "evaluation"], splits):
        save_jsonl(rows, args.output / (name + ".jsonl"))
    print({name: len(rows) for name, rows in zip(["demos", "feedback", "evaluation"], splits)})


if __name__ == "__main__":
    main()
