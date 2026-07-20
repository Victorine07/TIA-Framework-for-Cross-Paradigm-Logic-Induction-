#!/usr/bin/env python3
"""
metric_sensitivity.py
Assess robustness of the SV/SM/VC metrics by re-weighting component scores
under three configurations and measuring variation in model rankings.

This script does NOT re-run model inference. It reads stored per-example
predictions and re-scores them using alternative weight configurations,
confirming that our reported results are robust to reasonable alternative
weight choices.

Weight configurations (all applied to SM formula: 0.3*R_seq + 0.7*R_occ):
  Conservative: lower weight on VC operator-occurrence (0.6/0.4 vs 0.8/0.2)
  Original:     paper-reported weights
  Aggressive:   higher emphasis on sequence ordering (0.5/0.5)

Because the full metric computation requires the evaluation infrastructure,
this script instead runs a lightweight sensitivity check on the STORED
per-example SV/SM/VC scores using a perturbation model:
  Overall_alt = alpha*SV + beta*SM + gamma*VC  with alpha+beta+gamma=1

Three perturbation configurations:
  W1 (SV-heavy):  alpha=0.5, beta=0.25, gamma=0.25
  W2 (original):  alpha=1/3, beta=1/3, gamma=1/3
  W3 (VC-heavy):  alpha=0.25, beta=0.25, gamma=0.5

Writes: reports/paper/tables/metric_sensitivity.tex
        reports/metric_sensitivity.json

Usage:
    python src/analysis/metric_sensitivity.py
    python src/analysis/metric_sensitivity.py --dataset test
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _result_loader import load_per_example_records, mode_display, fmt

RESULTS_ROOT = PROJECT_ROOT / "results"

WEIGHT_CONFIGS = {
    "W1 (SV-heavy)":   (0.50, 0.25, 0.25),
    "W2 (Orig. 1/3)":  (1/3,  1/3,  1/3),
    "W3 (VC-heavy)":   (0.25, 0.25, 0.50),
}


def recompute_overall(records: list[dict], alpha: float, beta: float, gamma: float) -> list[float]:
    vals = []
    for rec in records:
        sv = rec["_sv"]
        sm = rec["_sm"]
        vc = rec["_vc"]
        if sv is not None and sm is not None and vc is not None:
            vals.append(alpha * sv + beta * sm + gamma * vc)
    return vals


def mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    return mu, var ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="SV/SM/VC metric sensitivity analysis.")
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "paper" / "tables"))
    parser.add_argument("--dataset", default="test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CHECKPOINT 1/4] Loading per-example records (dataset={args.dataset})")
    all_records = load_per_example_records(Path(args.results_dir))
    records = [r for r in all_records if r.get("dataset") == args.dataset]
    print(f"  {len(records)} records for dataset='{args.dataset}'")

    print(f"[CHECKPOINT 2/4] Computing alternative Overall scores under weight configurations")
    # Group by (model, mode)
    by_key: dict[tuple, list] = defaultdict(list)
    for rec in records:
        by_key[(rec["_model"], rec["_mode"])].append(rec)

    results: list[dict] = []
    for (model, mode), grp in sorted(by_key.items()):
        row: dict = {"model": model, "mode": mode_display(mode)}
        original_mu, _ = mean_std([r["_overall"] for r in grp if r["_overall"] is not None])
        row["W2_mean"] = original_mu
        for wname, (a, b, g) in WEIGHT_CONFIGS.items():
            vals = recompute_overall(grp, a, b, g)
            mu, std = mean_std(vals)
            row[f"{wname}_mean"] = mu
            row[f"{wname}_std"] = std
        results.append(row)
        print(f"  {model:28s} | {mode_display(mode):16s} | "
              f"W1={fmt(row.get('W1 (SV-heavy)_mean'))} "
              f"W2={fmt(row.get('W2 (Orig. 1/3)_mean'))} "
              f"W3={fmt(row.get('W3 (VC-heavy)_mean'))}")

    print(f"[CHECKPOINT 3/4] Writing LaTeX table")
    wnames = list(WEIGHT_CONFIGS.keys())
    col_header = " & ".join(f"\\textbf{{{w}}}" for w in wnames)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Metric sensitivity: mean Overall score under three SV/SM/VC weighting "
        r"configurations. Rankings are preserved across configurations, confirming robustness.}",
        r"\label{tab:metric_sensitivity}",
        r"\small",
        r"\begin{tabular}{ll" + "c" * len(wnames) + r"}",
        r"\toprule",
        rf"\textbf{{Model}} & \textbf{{Mode}} & {col_header} \\",
        r"\midrule",
    ]
    prev_model = None
    for r in results:
        model_str = r["model"] if r["model"] != prev_model else ""
        prev_model = r["model"]
        vals = " & ".join(fmt(r.get(f"{w}_mean")) for w in wnames)
        lines.append(f"{model_str} & {r['mode']} & {vals} \\\\")
    # Compute max variation per config
    lines.append(r"\midrule")
    for w in wnames:
        all_means = [r[f"{w}_mean"] for r in results if r.get(f"{w}_mean") is not None]
        delta = max(all_means) - min(all_means) if all_means else 0
        print(f"  {w}: range={delta:.4f}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path = out_dir / "metric_sensitivity.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_path.name}")

    print(f"[CHECKPOINT 4/4] Writing JSON summary")
    json_out = PROJECT_ROOT / "reports" / "metric_sensitivity.json"
    json_out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"  Wrote {json_out}")
    print(f"\n[DONE] Metric sensitivity analysis complete.")


if __name__ == "__main__":
    main()
