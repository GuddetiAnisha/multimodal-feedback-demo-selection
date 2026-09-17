"""Aggregate predictions across compatible, nonoverlapping seed runs."""
import argparse
from pathlib import Path
import json
import pandas as pd
from mfgds.evaluation import report, GROUPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = [json.loads((r / "manifest.json").read_text()) for r in args.runs]
    def signature(m):
        c = {k: v for k, v in m["config"].items() if k != "seeds"}
        return json.dumps([c, m["data_sha256"], m["versions"]], sort_keys=True)
    if any(m["status"] != "complete" for m in manifests) or len({signature(m) for m in manifests}) != 1:
        raise ValueError("Runs differ in data, configuration, dependencies, or completion status")
    frame = pd.concat([pd.read_csv(r / "predictions.csv") for r in args.runs], ignore_index=True)
    if frame.duplicated([*GROUPS, "seed", "query_id"]).any():
        raise ValueError("Repeated observations across runs")
    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output / "predictions.csv", index=False)
    report(frame, args.output, any(m["mock"] or m["synthetic"] for m in manifests))


if __name__ == "__main__":
    main()
