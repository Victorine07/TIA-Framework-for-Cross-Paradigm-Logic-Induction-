#!/usr/bin/env python3
"""
tier_metrics_summary.py
Compute tier-wise SV / SM / VC for every (model, mode, strategy, dataset) combination.

Reads:  results/**/*.jsonl  (per-example prediction files)
Writes: reports/paper/tables/tier_metrics_{dataset}.tex
        reports/tier_metrics_full.csv
        reports/tier_metrics_full.json

Usage:
    python src/analysis/tier_metrics_summary.py
    python src/analysis/tier_metrics_summary.py --dataset test
    python src/analysis/tier_metrics_summary.py --dataset test --mode zero_shot
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _result_loader import (
    load_per_example_records, model_short, mode_canonical, mode_display, fmt
)

RESULTS_ROOT = PROJECT_ROOT / "results"
TIERS = ["T1", "T2", "T3", "T4"]
METRICS = ["sv", "sm", "vc", "overall"]


def aggregate_tier_metrics(records: list[dict]) -> dict:
    """
    Aggregate records by (model, mode, strategy, dataset, tier).
    Returns nested dict: {(model,mode,strategy,dataset): {tier: {metric: [values]}}}
    """
    bucket: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        key = (rec["_model"], rec["_mode"], rec["_strategy"], rec.get("dataset", "?"))
        tier = rec["_tier"]
        for metric in METRICS:
            val = rec.get(f"_{metric}")
            if val is not None:
                bucket[key][tier][metric].append(float(val))
    return bucket


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def build_rows(bucket: dict) -> list[dict]:
    rows = []
    for (model, mode, strategy, dataset), tier_data in sorted(bucket.items()):
        base = {
            "model": model,
            "mode": mode,
            "mode_display": mode_display(mode),
            "strategy": strategy,
            "dataset": dataset,
        }
        for tier in TIERS:
            td = tier_data.get(tier, {})
            row = dict(base)
            row["tier"] = tier
            row["n"] = len(td.get("sv", []))
            for metric in METRICS:
                row[metric] = mean(td.get(metric, []))
            rows.append(row)
    return rows


def tex_tier_table(rows: list[dict], dataset: str, out: Path) -> None:
    """
    Write a LaTeX table of tier-wise metrics for a given dataset,
    grouped by model and mode.
    """
    filtered = [r for r in rows if r["dataset"] == dataset]
    if not filtered:
        print(f"  [WARNING] No rows for dataset={dataset}")
        return

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Tier-wise SV / SM / VC on \texttt{{{dataset}}} split. Best per model/mode/tier in \textbf{{bold}}.}}",
        rf"\label{{tab:tier_metrics_{dataset}}}",
        r"\small",
        r"\begin{tabular}{llllcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Mode} & \textbf{Strategy} & \textbf{Tier} "
        r"& \textbf{SV} & \textbf{SM} & \textbf{VC} & \textbf{Overall} \\",
        r"\midrule",
    ]

    prev_model = prev_mode = None
    for r in sorted(filtered, key=lambda x: (x["model"], x["mode"], x["strategy"], x["tier"])):
        model_str = r["model"] if r["model"] != prev_model else ""
        mode_str = r["mode_display"] if (r["model"], r["mode"]) != (prev_model, prev_mode) else ""
        prev_model = r["model"]
        prev_mode = r["mode"]
        lines.append(
            f"{model_str} & {mode_str} & {r['strategy']} & {r['tier']} "
            f"& {fmt(r['sv'])} & {fmt(r['sm'])} & {fmt(r['vc'])} & {fmt(r['overall'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tier-wise SV/SM/VC from per-example results.")
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "paper" / "tables"))
    parser.add_argument("--dataset", default=None, help="Filter to one dataset (e.g. test)")
    parser.add_argument("--mode", default=None, help="Filter to one mode (e.g. zero_shot)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_dir)

    print(f"[CHECKPOINT 1/5] Loading per-example records from: {results_root}")
    records = load_per_example_records(results_root)
    print(f"  Loaded {len(records)} per-example records")

    if args.dataset:
        records = [r for r in records if r.get("dataset") == args.dataset]
        print(f"  After dataset filter ({args.dataset}): {len(records)} records")
    if args.mode:
        records = [r for r in records if r["_mode"] == args.mode]
        print(f"  After mode filter ({args.mode}): {len(records)} records")

    print(f"[CHECKPOINT 2/5] Aggregating tier-wise metrics")
    bucket = aggregate_tier_metrics(records)
    print(f"  {len(bucket)} unique (model, mode, strategy, dataset) combinations")

    print(f"[CHECKPOINT 3/5] Building result rows")
    rows = build_rows(bucket)
    print(f"  Built {len(rows)} tier-metric rows")

    print(f"[CHECKPOINT 4/5] Writing LaTeX tables")
    datasets = sorted(set(r["dataset"] for r in rows))
    for ds in datasets:
        tex_out = out_dir / f"tier_metrics_{ds}.tex"
        tex_tier_table(rows, ds, tex_out)
        print(f"  Wrote {tex_out.name}")

    print(f"[CHECKPOINT 5/5] Writing CSV and JSON")
    csv_path = PROJECT_ROOT / "reports" / "tier_metrics_full.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "mode", "mode_display", "strategy", "dataset", "tier", "n",
                  "sv", "sm", "vc", "overall"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {csv_path}")

    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"  Wrote {json_path}")
    print(f"\n[DONE] Tier metrics written to: {out_dir}")


if __name__ == "__main__":
    main()
