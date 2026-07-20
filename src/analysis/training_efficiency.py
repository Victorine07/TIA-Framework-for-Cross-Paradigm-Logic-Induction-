#!/usr/bin/env python3
"""
training_efficiency.py
Summarize training configuration, resource usage, and efficiency metrics
from experiment run configs stored in overall summary JSON files.

Reads:  results/**/overall__*.json
Writes: reports/paper/tables/training_efficiency.tex
        reports/training_efficiency.json

Usage:
    python src/analysis/training_efficiency.py
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analysis"))
from _result_loader import load_overall_summaries, mode_display, fmt

RESULTS_ROOT = PROJECT_ROOT / "results"


def extract_config_row(row: dict) -> dict | None:
    """Extract efficiency-relevant fields from one overall summary row."""
    rc = row.get("run_config", {})
    if not rc:
        return None
    mode = row["mode"]
    # Quantization: load_in_4bit is how our eval runner records it
    quant = rc.get("load_in_4bit")
    quant_str = "NF4 4-bit" if quant else ("FP16" if quant is False else "---")
    # GPU name
    cuda_info = rc.get("cuda", {})
    gpu_names = cuda_info.get("device_names", [])
    gpu = gpu_names[0] if gpu_names else "---"
    model_class = rc.get("model_class", "---")
    return {
        "model": row["model"],
        "mode": mode_display(mode),
        "strategy": row["strategy"],
        "dataset": row["dataset"],
        "quantization": quant_str,
        "model_class": model_class,
        "batch_size": rc.get("batch_size", "---"),
        "max_new_tokens": rc.get("max_new_tokens", "---"),
        "max_input_length": rc.get("max_input_length", "---"),
        "gpu": gpu,
        "total_time_sec": row.get("total_time_sec"),
    }


def fmt_config(val) -> str:
    if val is None or val == "---":
        return "---"
    if isinstance(val, float):
        return f"{val:.2e}" if val < 0.001 else f"{val:.4f}"
    return str(val)


def tex_training_table(config_rows: list[dict], out: Path) -> None:
    """Write inference configuration table for each mode × model."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inference evaluation configuration by model and regime. "
        r"All runs: greedy decoding ($T{=}0$), Tesla V100-PCIE-16GB.}",
        r"\label{tab:training_efficiency}",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Mode} & \textbf{Quant.} "
        r"& \textbf{Max Tokens} & \textbf{Eval Time (s)} \\",
        r"\midrule",
    ]
    # Deduplicate to one representative row per (model, mode, strategy=none, dataset=test)
    seen: set = set()
    for r in sorted(config_rows, key=lambda x: (x["model"], x["mode"], x["strategy"])):
        if r["dataset"] != "test" or r["strategy"] not in ("none", "full", "alljson"):
            continue
        key = (r["model"], r["mode"])
        if key in seen:
            continue
        seen.add(key)
        t = r["total_time_sec"]
        t_str = f"{int(t)}" if t else "---"
        lines.append(
            f"{r['model']} & {r['mode']} & {r['quantization']} "
            f"& {fmt_config(r['max_new_tokens'])} & {t_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize training efficiency from run configs.")
    parser.add_argument("--results-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "paper" / "tables"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CHECKPOINT 1/4] Loading overall summaries from: {args.results_dir}")
    rows = load_overall_summaries(Path(args.results_dir))
    print(f"  {len(rows)} dataset-level rows")

    print(f"[CHECKPOINT 2/4] Extracting run configuration fields")
    config_rows = []
    for r in rows:
        cr = extract_config_row(r)
        if cr:
            config_rows.append(cr)
    print(f"  {len(config_rows)} rows with run_config data")

    # Print summary to stdout
    for r in sorted(config_rows, key=lambda x: (x["model"], x["mode"], x["strategy"], x["dataset"])):
        print(
            f"  {r['model']:28s} | {r['mode']:16s} | {r['strategy']:12s} | "
            f"quant={r['quantization']} batch={r['batch_size']} "
            f"time={fmt(r.get('total_time_sec'))}s"
        )

    print(f"[CHECKPOINT 3/4] Writing LaTeX table")
    tex_training_table(config_rows, out_dir / "training_efficiency.tex")
    print(f"  Wrote training_efficiency.tex")

    print(f"[CHECKPOINT 4/4] Writing JSON summary")
    json_out = PROJECT_ROOT / "reports" / "training_efficiency.json"
    json_out.write_text(json.dumps(config_rows, indent=2, default=str), encoding="utf-8")
    print(f"  Wrote {json_out}")
    print(f"\n[DONE] Training efficiency analysis complete.")


if __name__ == "__main__":
    main()
