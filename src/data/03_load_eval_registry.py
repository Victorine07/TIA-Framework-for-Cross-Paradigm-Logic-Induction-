#!/usr/bin/env python3
"""
03_load_eval_registry.py

Simple inspection script for the evaluation registry.
Use this to verify that the registry JSON was created correctly
before training or inference scripts consume it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def resolve_default_registry(project_root: Path) -> Path:
    candidates = [
        project_root / "datasets" / "processed" / "eval_registry_family_holdout_v1.json",
        project_root / "datasets" / "processed" / "eval_registry.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_keys(obj: Any) -> List[str]:
    if isinstance(obj, dict):
        return sorted(obj.keys())
    return []


def infer_project_root(registry_path: Path, registry: Dict[str, Any]) -> Path:
    # Deliberately ignore any stored project-root field in the registry --
    # it would be a snapshot of wherever the registry was built (e.g. a
    # local dev machine), not where it currently lives. Always derive from
    # this registry file's actual on-disk location instead, so the same
    # registry JSON prints the right root after being copied to the
    # cluster. See src/data/registry_loader.py for the same fix applied to
    # the bug this masked: this function currently never hits the stored
    # value because the live registry schema uses "project_root_abs", not
    # "project_root" -- but it's the same trust-stored-path-over-derive-it
    # pattern, just not yet armed, so it's fixed here too.
    _ = registry
    return registry_path.resolve().parents[2]


def print_header(title: str, width: int = 72) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


def summarize_registry(registry_path: Path, registry: Dict[str, Any]) -> None:
    project_root = infer_project_root(registry_path, registry)

    tag = registry.get("tag", "unknown")
    version = registry.get("version", "unknown")
    datasets = registry.get("datasets", {})
    groups = registry.get("groups", {})
    counts = registry.get("counts", {})

    print_header("Registry Loader Summary")
    print(f"Registry path: {registry_path}")
    print(f"Project root:  {project_root}")
    print(f"Tag:           {tag}")
    print(f"Version:       {version}")
    print(f"Datasets:      {', '.join(safe_keys(datasets)) if datasets else '(none)'}")
    print(f"Groups:        {', '.join(safe_keys(groups)) if groups else '(none)'}")
    print(f"Counts:        {counts if counts else '{}'}")
    print()

    if datasets:
        print_header("Dataset Details")
        for name in safe_keys(datasets):
            entry = datasets[name]
            if isinstance(entry, dict):
                path = entry.get("path", "(missing path)")
                n = entry.get("num_examples", entry.get("count", "?"))
                desc = entry.get("description", "")
                print(f"- {name}")
                print(f"  path: {path}")
                print(f"  examples: {n}")
                if desc:
                    print(f"  description: {desc}")
            else:
                print(f"- {name}: {entry}")
        print()

    if groups:
        print_header("Group Details")
        for name in safe_keys(groups):
            entry = groups[name]
            if isinstance(entry, dict):
                members = entry.get("members", entry.get("datasets", []))
                desc = entry.get("description", "")
                print(f"- {name}")
                print(f"  members: {members}")
                if desc:
                    print(f"  description: {desc}")
            else:
                print(f"- {name}: {entry}")
        print()


def validate_registry(registry: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    required_top_level = ["datasets", "groups", "counts"]
    for key in required_top_level:
        if key not in registry:
            issues.append(f"Missing top-level key: '{key}'")

    datasets = registry.get("datasets")
    if datasets is not None and not isinstance(datasets, dict):
        issues.append("'datasets' must be a dictionary")

    groups = registry.get("groups")
    if groups is not None and not isinstance(groups, dict):
        issues.append("'groups' must be a dictionary")

    counts = registry.get("counts")
    if counts is not None and not isinstance(counts, dict):
        issues.append("'counts' must be a dictionary")

    return issues


def main() -> None:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]

    parser = argparse.ArgumentParser(description="Load and inspect an evaluation registry JSON.")
    parser.add_argument(
        "--registry",
        type=str,
        default=str(resolve_default_registry(project_root)),
        help="Path to eval registry JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if validation issues are found",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry).expanduser().resolve()

    try:
        registry = load_json(registry_path)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    issues = validate_registry(registry)
    summarize_registry(registry_path, registry)

    if issues:
        print_header("Validation Issues")
        for issue in issues:
            print(f"- {issue}")
        if args.strict:
            sys.exit(1)
    else:
        print_header("Validation")
        print("Registry structure looks OK.")


if __name__ == "__main__":
    main()