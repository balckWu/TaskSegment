"""Summarize exactly three formal runs with matching protocols."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs=3, help="metrics.json for seeds 42, 3407, 2026")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.metrics]
    if {r["seed"] for r in runs} != {42, 3407, 2026} or any(r["smoke"] for r in runs):
        raise ValueError("Need three formal runs with distinct seeds 42, 3407, 2026")
    keys = ("split_fingerprint", "source_hashes", "model_config", "eval_support_count",
            "metric_size", "threshold", "postprocess")
    for key in keys:
        if any(r[key] != runs[0][key] for r in runs):
            raise ValueError(f"Incompatible {key}; do not pool these runs")
    ignored = {"output", "seed", "device", "resume", "cache_dataset", "eval_batch_size"}
    protocols = [{k: v for k, v in r["training_protocol"].items() if k not in ignored} for r in runs]
    if any(p != protocols[0] for p in protocols):
        raise ValueError("Training protocols differ across seeds")
    def summarize(values):
        return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "seed_values": values}
    result = {"seeds": [r["seed"] for r in runs], "eval_support_count": runs[0]["eval_support_count"],
        "macro": {m: summarize([r["macro"][m] for r in runs]) for m in ("dice", "iou", "hd95")},
        "per_domain": {d: {m: summarize([r["per_domain"][d][m] for r in runs])
                            for m in ("dice", "iou", "hd95")} for d in runs[0]["per_domain"]}}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for metric, value in result["macro"].items():
        print(f"Macro {metric}: {value['mean']:.5f} +/- {value['sample_std']:.5f}")


if __name__ == "__main__":
    main()
