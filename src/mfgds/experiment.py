"""Reproducible feedback collection, training, ablations and held-out evaluation."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
import yaml
from .checkpoints import Checkpoints, atomic_bytes, output_lock
from .devices import resolve_device
from .data import synthetic, load_jsonl, validate_splits
from .embeddings import HashEncoder, ClipEncoder, retrieval_view
from .evaluation import score, report
from .models import MockLMM, HuggingFaceLMM
from .retrievers import VectorIndex, UtilityRetriever, ExternalGrip, pair_features
from .selectors import select, order, fit_context, diversity

METHODS = {"random", "visual", "textual", "multimodal", "feedback", "grip_approx", "grip_external"}
CHANNELS = {"full": ("image", "question", "answer"), "no_answer": ("image", "question"),
            "no_image": ("question", "answer"), "no_question": ("image", "answer"),
            "no_diversity": ("image", "question", "answer"), "no_feedback": ("image", "question", "answer")}


def run(config, output, resume=False):
    with output_lock(output):
        return _run(config, output, resume)


def _run(config, output, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and ((output / "manifest.json").exists() or (output / "checkpoints").exists()):
        raise FileExistsError("Use a fresh output directory to preserve prior results")
    device = resolve_device(config.get("device", "cuda"))
    print(f"Compute device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""), flush=True)
    index_backend = config.get("index_backend", "torch" if device.type == "cuda" else "auto")
    if device.type == "cuda" and index_backend != "torch":
        raise ValueError("GPU mode requires index_backend: torch (or --device cuda)")
    methods = config.get("methods", sorted(METHODS - {"grip_external"}))
    variants = config.get("variants", ["full"])
    if set(methods)-METHODS or set(variants)-set(CHANNELS):
        raise ValueError("Unknown method or ablation")
    if not config["ks"] or any(k < 0 for k in config["ks"]):
        raise ValueError("Nonempty nonnegative ks required")
    if not methods or not variants or not config.get("seeds", [0]):
        raise ValueError("Nonempty methods, variants, and seeds required")
    if not config["orderings"] or set(config["orderings"]) - {"best_first", "best_last", "random"}:
        raise ValueError("Invalid orderings")
    if not config["context_budgets"] or min(config["context_budgets"]) <= 0:
        raise ValueError("Positive context budgets required")
    for key in ["feedback_candidates", "candidate_pool", "epochs"]:
        if config.get(key, 1) < 1:
            raise ValueError(f"{key} must be positive")
    # Duplicate grid values would duplicate observations even in a fresh run.
    for key in ["methods", "variants", "seeds", "ks", "orderings", "context_budgets"]:
        values = config.get(key, [])
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {key} values")
    checkpoints = Checkpoints(output)
    state = checkpoints.load() if resume else None
    torch.set_num_threads(config.get("threads", 1))
    dataset = config.get("dataset", {"kind": "synthetic"})
    if dataset["kind"] == "synthetic":
        demos, train, test = synthetic(output / "fixtures", config.get("data_seed", 42), tuple(dataset.get("sizes", [24, 12, 12])))
    elif dataset["kind"] == "jsonl":
        demos, train, test = [load_jsonl(dataset[key]) for key in ["demos", "feedback", "evaluation"]]
    else:
        raise ValueError("Use synthetic or prepared jsonl dataset")
    validate_splits(demos, train, test)
    ecfg = config.get("encoder", {"kind": "hash"})
    if ecfg["kind"] not in {"hash", "clip"}:
        raise ValueError("Unknown encoder")
    encoder_device = str(device)
    encoder = HashEncoder() if ecfg["kind"] == "hash" else ClipEncoder(ecfg.get("model", "clip-ViT-B-32"), encoder_device)
    mcfg = config.get("model", {"kind": "mock"})
    if mcfg["kind"] not in {"mock", "hf"}:
        raise ValueError("Unknown model")
    mcfg = dict(mcfg, device=str(device))
    model = MockLMM(device=device) if mcfg["kind"] == "mock" else HuggingFaceLMM(**{k: v for k, v in mcfg.items() if k != "kind"})
    manifest = {"config": config, "status": "running", "mock": mcfg["kind"] == "mock", "synthetic": dataset["kind"] == "synthetic",
                "compute_device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "token_measurement": model.token_measurement, "python": platform.python_version(),
                "versions": {p: importlib.metadata.version(p) for p in ["torch", "numpy", "scikit-learn", "pandas", "Pillow"]},
                "split_ids": {k: [x.id for x in v] for k, v in [("demos", demos), ("feedback", train), ("evaluation", test)]}}
    # Include content fingerprints so replacing data under a filename is detectable.
    manifest["data_sha256"] = hashlib.sha256(json.dumps([
        [x.id, x.question, x.answer, x.choices, x.answers, x.group, hashlib.sha256(Path(x.image).read_bytes()).hexdigest() if x.image else None]
        for x in demos+train+test], sort_keys=True).encode()).hexdigest()
    identity = {"config": config, "data_sha256": manifest["data_sha256"], "versions": manifest["versions"], "python": manifest["python"]}
    identity["compute"] = {"device": str(device), "gpu": manifest["gpu_name"], "cuda": torch.version.cuda}
    if "grip_scores" in config:
        identity["grip_sha256"] = hashlib.sha256(Path(config["grip_scores"]).read_bytes()).hexdigest()
    if state is not None and state["identity"] != identity:
        raise ValueError("Resume configuration, data or dependency versions differ from checkpoint")
    if state is not None:
        print(f"Resuming valid checkpoint {checkpoints.sequence}: {state['progress']}; "
              f"skipping {len(state['rows'])} committed evaluation units", flush=True)
    if state is None:
        state = {"identity": identity, "rows": {}, "feedback": {}, "training": {}, "zero": {}, "progress": {}}
    def persist(**progress):
        state["progress"] = progress
        checkpoints.save(state)
    persist(stage="initializing")
    atomic_bytes(output / "manifest.json", json.dumps(manifest, indent=2).encode())
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    # Features encode demonstration answers; both query splits are answer-masked.
    encoded = {v: (encoder.encode(demos, channels=CHANNELS[v]),
                   encoder.encode([x.query() for x in train], query=True, channels=CHANNELS[v]),
                   encoder.encode([x.query() for x in test], query=True, channels=CHANNELS[v])) for v in variants}
    for x in train + test:
        fit_context(x.query(), [], model, min(config["context_budgets"]))
    def zero_scores(split, examples):
        values = []
        for x in examples:
            key = (split, x.id)
            if key not in state["zero"]:
                state["zero"][key] = score(model.predict(x.query(), []).text, x)
                persist(stage="zero_shot", split=split, query=x.id)
            values.append(state["zero"][key])
        return values
    train_zero = zero_scores("feedback", train)
    test_zero = zero_scores("evaluation", test)
    rows, feedback_rows, losses = [], [], []
    external = ExternalGrip(config["grip_scores"]) if "grip_external" in methods else None
    for seed in config.get("seeds", [0]):
        rng = np.random.default_rng(seed)
        pairs, rewards = [], []
        # Sample uniformly to retain positive, zero, and negative feedback; no test labels.
        for qi, query in enumerate(train):
            for di in rng.choice(len(demos), size=min(len(demos), config.get("feedback_candidates", 8)), replace=False):
                feedback_key = (seed, qi, int(di))
                if feedback_key in state["feedback"]:
                    cached = state["feedback"][feedback_key]
                    pairs.append((qi, int(di)))
                    rewards.append(cached["reward"])
                    feedback_rows.append(cached)
                    continue
                selected, _ = fit_context(query.query(), [demos[di]], model, config.get("feedback_context_budget", max(config["context_budgets"])))
                if not selected:
                    raise ValueError("Feedback context budget cannot fit one demonstration")
                pred = model.predict(query.query(), selected)
                value = score(pred.text, query)
                reward = value-train_zero[qi]
                pairs.append((qi, int(di)))
                rewards.append(reward)
                feedback_rows.append({"seed": seed, "query_id": query.id, "demo_id": demos[di].id,
                                      "zero_score": train_zero[qi], "demo_score": value, "reward": reward, "prediction": pred.text})
                state["feedback"][feedback_key] = feedback_rows[-1]
                persist(stage="feedback", seed=seed, query=query.id, demo=demos[di].id,
                        generator=rng.bit_generator.state)
        for variant in variants:
            d, qtrain, qtest = encoded[variant]
            trained = {}
            for learned in sorted(set(methods) & {"feedback", "grip_approx"}):
                ds, qs = (d[:, :encoder.dim], qtrain[:, :encoder.dim]) if learned == "grip_approx" else (d, qtrain)
                net = UtilityRetriever(ds.shape[1], seed, device=device)
                features = np.stack([pair_features(qs[qi], ds[di]) for qi, di in pairs])
                training_key = (seed, variant, learned)
                def save_epoch(payload):
                    state["training"][training_key] = payload
                    persist(stage="training", seed=seed, variant=variant, method=learned,
                            epoch=payload["epoch"], generator=rng.bit_generator.state)
                history = net.fit(features, rewards, epochs=config.get("epochs", 60),
                                  resume_state=state["training"].get(training_key), checkpoint=save_epoch)
                net.save(output / f"retriever_{variant}_{learned}_{seed}.pt")
                losses.extend({"variant": variant, "method": learned, "seed": seed, "epoch": i, "mse": value} for i, value in enumerate(history))
                trained[learned] = net
            indexes = {m: VectorIndex(retrieval_view(d, m), index_backend, device=device)
                       for m in ["visual", "textual", "multimodal"]}
            manifest["index_backend"] = indexes["multimodal"].backend
            for qi, example in enumerate(test):
                query = example.query()
                for method in methods:
                    started = time.perf_counter()
                    q = qtest[qi]
                    # Fixed candidate cap; random baseline samples the whole pool.
                    actual_method = "multimodal" if variant == "no_feedback" and method == "feedback" else method
                    if actual_method == "random":
                        ids = rng.permutation(len(demos))[:config.get("candidate_pool", 32)]
                        scores = rng.random(len(ids))
                    else:
                        search_method = actual_method if actual_method in indexes else ("visual" if actual_method == "grip_approx" else "multimodal")
                        ids, scores = indexes[search_method].search(retrieval_view(q, search_method), config.get("candidate_pool", 32))
                        if actual_method in trained:
                            qs, ds = (q[:encoder.dim], d[ids, :encoder.dim]) if actual_method == "grip_approx" else (q, d[ids])
                            scores = trained[actual_method].score(qs, ds)
                        elif actual_method == "grip_external":
                            scores = external.score(example.id, [demos[i].id for i in ids])
                    retrieval_time = time.perf_counter()-started
                    for k, ordering, budget in itertools.product(config["ks"], config["orderings"], config["context_budgets"]):
                        strength = config.get("redundancy_penalty", .25) if actual_method in {"feedback", "grip_approx", "grip_external"} and variant != "no_diversity" else 0
                        selection_vectors = d[ids, :encoder.dim] if actual_method == "grip_approx" else d[ids]
                        local = select(scores, selection_vectors, k, strength, device=device)
                        local = order(local, scores, ordering, rng)
                        selected_ids = [int(ids[i]) for i in local]
                        kept, tokens = fit_context(query, [demos[i] for i in selected_ids], model, budget)
                        selected_ids = selected_ids[:len(kept)]
                        unit = (seed, variant, qi, method, k, ordering, budget)
                        # Replay cheap selection RNG draws in original order, skip inference.
                        if unit in state["rows"]:
                            rows.append(state["rows"][unit])
                            continue
                        pred = model.predict(query, kept)
                        value = score(pred.text, example)
                        div, red = diversity(d[selected_ids], device=device)
                        rows.append({"seed": seed, "variant": variant, "method": method, "k": k, "ordering": ordering,
                                     "context_budget": budget, "query_id": example.id, "query_group": example.group or example.id,
                                     "prediction": pred.text, "target": example.answer, "score": value, "zero_score": test_zero[qi],
                                     "gain": value-test_zero[qi], "effective_k": len(kept), "demo_ids": json.dumps([x.id for x in kept]),
                                     "latency_s": pred.seconds, "retrieval_s": retrieval_time, "input_tokens": pred.input_tokens,
                                     "output_tokens": pred.output_tokens, "diversity": div, "redundancy": red,
                                     "predicted_utility": float(np.mean(scores[local[:len(kept)]])) if kept and actual_method in trained else None})
                        state["rows"][unit] = rows[-1]
                        persist(stage="evaluation", seed=seed, variant=variant, method=method,
                                k=k, ordering=ordering, context_budget=budget, query=example.id,
                                generator=rng.bit_generator.state)
        print(f"Completed seed {seed}: {len(rows)} prediction rows", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(feedback_rows).to_csv(output / "feedback.csv", index=False)
    feedback_frame = pd.DataFrame(feedback_rows)
    feedback_frame.assign(positive=feedback_frame.reward.gt(0), negative=feedback_frame.reward.lt(0)).groupby("seed").agg(
        mean_reward=("reward", "mean"), positive_fraction=("positive", "mean"),
        negative_fraction=("negative", "mean"), pairs=("reward", "size")).to_csv(output / "feedback_summary.csv")
    pd.DataFrame(losses).to_csv(output / "training.csv", index=False)
    report(frame, output, manifest["mock"] or manifest["synthetic"])
    manifest["status"] = "complete"
    manifest["prediction_rows"] = len(rows)
    persist(stage="complete", prediction_rows=len(rows))
    atomic_bytes(output / "manifest.json", json.dumps(manifest, indent=2).encode())
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scienceqa.yaml")
    parser.add_argument("--output", default="results/scienceqa_gpu")
    parser.add_argument("--resume", action="store_true", help="Continue the latest valid checkpoint in --output")
    parser.add_argument("--device", help="Override all model/training/retrieval placement: cuda, cuda:N, cpu, auto")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
        config["index_backend"] = "torch"
    try:
        run(config, args.output, resume=args.resume)
    except KeyboardInterrupt:
        print("Stopped. Resume with the same --config and --output plus --resume.", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
