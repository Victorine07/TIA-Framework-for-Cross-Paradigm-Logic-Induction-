"""
_result_loader.py
Shared utility: discover, deduplicate, and load experiment result files.

All analysis scripts import from this module so deduplication logic lives in one place.
Deduplication rule: for each (model, mode, strategy, dataset) key, keep the newest run
(determined by timestamp embedded in the filename).
"""

from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

# ── Normalization helpers ────────────────────────────────────────────────────

def model_short(model_id: str) -> str:
    m = model_id.lower()
    if "qwen2.5-coder-7b" in m:
        return "Qwen2.5-7B"
    if "deepseek-coder-v2-lite" in m:
        return "DS-Coder-V2-Lite"
    return model_id.split("/")[-1][:28]


def mode_canonical(mode_raw: str) -> str:
    """Normalize diverse mode tag strings to a canonical form."""
    m = mode_raw.lower().replace("-", "_")
    if m in ("zeroshot", "zero_shot"):
        return "zero_shot"
    if re.match(r"fewshot_k\d+", m):
        return m          # already canonical: fewshot_k3, fewshot_k5
    if m in ("finetuned", "sft", "finetune"):
        return "finetuned"
    return mode_raw


def mode_display(mode_canonical: str) -> str:
    if mode_canonical == "zero_shot":
        return "Zero-shot"
    if mode_canonical == "finetuned":
        return "SFT"
    m = re.match(r"fewshot_k(\d+)", mode_canonical)
    if m:
        return f"Few-shot k={m.group(1)}"
    return mode_canonical


# ── File discovery ───────────────────────────────────────────────────────────

def _ts_from_path(p: Path) -> str:
    """Extract YYYYMMDD_HHMMSS timestamp from filename for dedup sorting."""
    m = re.search(r"(\d{8}_\d{6})", p.stem)
    return m.group(1) if m else "00000000_000000"


def find_overall_summaries(results_root: Path = RESULTS_ROOT) -> list[Path]:
    return sorted(results_root.rglob("overall__*.json"))


def find_per_example_jsonls(results_root: Path = RESULTS_ROOT) -> list[Path]:
    """Return all per-example JSONL files (exclude __summary.json files)."""
    return [
        p for p in sorted(results_root.rglob("*.jsonl"))
        if not p.name.endswith("__summary.json")
    ]


def _overall_key(p: Path) -> tuple:
    """Derive (model, mode, strategy) from overall summary filename."""
    # overall__<model>__<mode>__<strategy>__<ts>.json
    parts = p.stem.split("__")
    if len(parts) >= 4:
        return (parts[1], parts[2], parts[3])
    return (p.stem,)


def _perex_key(p: Path) -> tuple:
    """Derive (dataset, model, mode, strategy) from per-example JSONL filename."""
    # <dataset>__<model>__<mode>__<strategy>__<ts>.jsonl
    parts = p.stem.split("__")
    if len(parts) >= 4:
        return (parts[0], parts[1], parts[2], parts[3])
    return (p.stem,)


def deduplicate_newest(paths: list[Path], keyfn) -> list[Path]:
    """For each logical key, keep the path with the newest timestamp."""
    by_key: dict[tuple, Path] = {}
    for p in paths:
        key = keyfn(p)
        if key not in by_key or _ts_from_path(p) > _ts_from_path(by_key[key]):
            by_key[key] = p
    return sorted(by_key.values())


# ── Loading helpers ──────────────────────────────────────────────────────────

def load_overall_summaries(results_root: Path = RESULTS_ROOT) -> list[dict]:
    """
    Load all overall summary JSONs, deduplicated to newest run per key.
    Returns flat list of dataset-level rows with normalized fields.
    """
    all_paths = find_overall_summaries(results_root)
    deduped = deduplicate_newest(all_paths, _overall_key)
    rows = []
    for p in deduped:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARNING] Could not parse {p.name}: {e}")
            continue
        model_raw = d.get("base_model") or d.get("model") or "unknown"
        mode_raw = d.get("mode") or "unknown"
        strategy = d.get("metadata_strategy", "unknown")
        for ds_name, ds in d.get("datasets", {}).items():
            row = {
                "model_raw": model_raw,
                "model": model_short(model_raw),
                "mode_raw": mode_raw,
                "mode": mode_canonical(mode_raw),
                "strategy": strategy,
                "dataset": ds_name,
                "n": ds.get("count", 0),
                "avg_sv": ds.get("avg_sv"),
                "avg_sm": ds.get("avg_sm"),
                "avg_vc": ds.get("avg_vc"),
                "avg_overall": ds.get("avg_overall"),
                "by_tier": ds.get("by_tier", {}),
                "by_family": ds.get("by_family", {}),
                "by_cipher": ds.get("by_cipher", {}),
                "failure_counts": ds.get("failure_counts", {}),
                "run_config": d.get("run_config", {}),
                "total_time_sec": ds.get("total_time_sec"),
                "source_path": str(p),
            }
            rows.append(row)
    return rows


def load_per_example_records(
    results_root: Path = RESULTS_ROOT,
    datasets: list[str] | None = None,
) -> list[dict]:
    """
    Load all per-example JSONL records, deduplicated to newest run per
    (dataset, model, mode, strategy) key. Returns flat list of records
    with normalized top-level fields added.

    Each record has: model, mode, strategy, tier, family, cipher,
                     sv, sm, vc, overall, failure_category
    """
    all_paths = find_per_example_jsonls(results_root)
    deduped = deduplicate_newest(all_paths, _perex_key)
    records = []
    for p in deduped:
        if datasets:
            # quick dataset filter from filename prefix
            ds_prefix = p.stem.split("__")[0]
            if ds_prefix not in datasets:
                continue
        try:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    meta = rec.get("metadata", {})
                    metrics = rec.get("metrics", {})
                    model_raw = rec.get("model") or rec.get("base_model") or "unknown"
                    mode_raw = rec.get("mode") or "unknown"
                    # strategy from prompting block
                    strategy = (rec.get("prompting") or {}).get("metadata_strategy", "unknown")
                    rec["_model"] = model_short(model_raw)
                    rec["_mode"] = mode_canonical(mode_raw)
                    rec["_strategy"] = strategy
                    rec["_tier"] = meta.get("tier", "UNK")
                    rec["_family"] = meta.get("family", "UNK")
                    rec["_cipher"] = meta.get("cipher", "UNK")
                    rec["_sv"] = metrics.get("sv")
                    rec["_sm"] = metrics.get("sm")
                    rec["_vc"] = metrics.get("vc")
                    rec["_overall"] = metrics.get("overall")
                    rec["_failure"] = rec.get("failure_category", "unknown")
                    records.append(rec)
        except Exception as e:
            print(f"[WARNING] Could not load {p.name}: {e}")
    return records


def fmt(val, digits: int = 3) -> str:
    if val is None:
        return "---"
    return f"{float(val):.{digits}f}"
