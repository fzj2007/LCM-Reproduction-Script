#!/usr/bin/env python3
"""prepare_stats.py - derive the statistics files that the OFFICIAL LCM source
expects but that are not shipped in the OSF "runs" archive.

This script ONLY reads the official OSF artifacts under data/runs. All derived
compatibility files are written under prepared/ so that the downloaded OSF
runs directory stays byte-for-byte read-only (important when data/runs is
symlinked to an existing archive):
  1. prepared/statistics_complex_workload_combined.json
     (derived from parsed_plans/statistics_workload_combined.json + the
      workers_planned numeric feature observed in the official workloads;
      used by flat and dace)
  2. prepared/qpp_stats/<db>.json
     (a byte-identical copy of json/<db>/feature_statistics_combined.json;
      used by qppnet)
  3. prepared/statistics_zeroshot_train.json
     (combined statistics + any categorical values observed in the official
      workloads that are missing from the value_dicts, e.g. the bytea
      data_type on credit.member.photograph; used by zeroshot)

The official repository source code (lcm-eval/) is NEVER modified.
All paths are relative to the script location so the kit is portable.
"""

import argparse
import glob
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path


WORKERS_PATTERN = re.compile(r'"workers_planned"\s*:\s*(\d+(?:\.\d+)?)')

# categorical features used by PostgresEstSystemCardDetail (zeroshot family)
CATEGORICAL = {
    "data_type": "database_stats.column_stats[].data_type",
    "op_name": "plan_parameters.op_name",
    "operator": "filter_columns[].operator",
    "aggregation": "output_columns[].aggregation",
}

TARGET_DATABASES = ("imdb", "baseball", "tpc_h_pk")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_workers(path: Path):
    """Yield every workers_planned value found in a workload file (streamed)."""
    carry = ""
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            text = carry + chunk
            body, carry = text[:-128], text[-128:]
            yield from (float(match) for match in WORKERS_PATTERN.findall(body))
    yield from (float(match) for match in WORKERS_PATTERN.findall(carry))


def robust_scale(values):
    """RobustScaler-equivalent (quantile_range=(25,75)) without sklearn."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise RuntimeError("no values to scale")

    def quantile(q):
        pos = (n - 1) * q
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    center = quantile(0.5)
    scale = quantile(0.75) - quantile(0.25)
    if scale == 0:
        scale = 1.0
    return center, scale


def collect_categorical_values(parsed_root: Path):
    """Collect every categorical value observed in the official workloads."""
    values = {name: Counter() for name in CATEGORICAL}
    workload_paths = sorted(
        path for path in parsed_root.glob("*/workload_100k_s1_c8220.json")
        if path.parent.name != "tpc_h"
    )
    for path in workload_paths:
        run = json.load(open(path, encoding="utf-8"))
        dbstats = run.get("database_stats", {}) or {}
        for col in dbstats.get("column_stats", []) or []:
            if isinstance(col, dict) and "data_type" in col:
                values["data_type"][str(col["data_type"])] += 1

        def walk_plan(node):
            pp = node.get("plan_parameters", {}) or {}
            if "op_name" in pp:
                values["op_name"][str(pp["op_name"])] += 1
            for oc in pp.get("output_columns", []) or []:
                if isinstance(oc, dict) and "aggregation" in oc:
                    values["aggregation"][str(oc["aggregation"])] += 1
            walk_filters(pp.get("filter_columns"))
            for child in node.get("children", []) or []:
                walk_plan(child)

        def walk_filters(fc):
            if not isinstance(fc, dict):
                return
            if "operator" in fc:
                values["operator"][str(fc["operator"])] += 1
            for child in fc.get("children", []) or []:
                walk_filters(child)

        for plan in run.get("parsed_plans", []) or []:
            walk_plan(plan)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="data/runs directory")
    parser.add_argument("--prepared", type=Path, required=True, help="prepared/ directory")
    args = parser.parse_args()

    parsed_root = args.runs / "parsed_plans"
    prepared_root = args.prepared
    prepared_root.mkdir(parents=True, exist_ok=True)
    (prepared_root / "qpp_stats").mkdir(parents=True, exist_ok=True)
    report = {"steps": []}

    # ---- 1. parsed_plans/statistics_complex_workload_combined.json ----
    source_stats = parsed_root / "statistics_workload_combined.json"
    target_stats = prepared_root / "statistics_complex_workload_combined.json"
    if not target_stats.exists():
        if not source_stats.exists():
            raise SystemExit(
                f"missing official statistics file: {source_stats}. "
                "Run 'setup' first (downloads the OSF runs archive)."
            )
        workload_paths = sorted(
            path for path in parsed_root.glob("*/workload_100k_s1_c8220.json")
            if path.parent.name != "tpc_h"
        )
        values = [value for path in workload_paths for value in extract_workers(path)]
        if not values:
            raise RuntimeError("No workers_planned values found in official workloads")
        center, scale = robust_scale(values)
        statistics = json.load(open(source_stats, encoding="utf-8"))
        statistics["workers_planned"] = {
            "max": float(max(values)),
            "scale": float(scale),
            "center": float(center),
            "type": "numeric",
        }
        with target_stats.open("w", encoding="utf-8") as handle:
            json.dump(statistics, handle, ensure_ascii=False, separators=(",", ":"))
        report["steps"].append(
            {
                "action": "derive",
                "source": str(source_stats),
                "target": str(target_stats),
                "note": "added workers_planned numeric feature (RobustScaler-equivalent)",
                "workers_planned_count": len(values),
            }
        )
        print(f"derived {target_stats} (+workers_planned, n={len(values)})")
    else:
        print(f"present: {target_stats}")

    # ---- 2. prepared/qpp_stats/<db>.json (byte copy of json stats) ----
    for database in TARGET_DATABASES:
        source = args.runs / "json" / database / "feature_statistics_combined.json"
        target = prepared_root / "qpp_stats" / f"{database}.json"
        if not target.exists():
            if not source.exists():
                raise SystemExit(
                    f"missing official statistics file: {source}. Run 'setup' first."
                )
            shutil.copyfile(source, target)
            report["steps"].append(
                {
                    "action": "copy",
                    "source": str(source),
                    "source_sha256": sha256(source),
                    "target": str(target),
                    "target_sha256": sha256(target),
                }
            )
            print(f"copied {source} -> {target}")
        else:
            print(f"present: {target}")

    # ---- 3. prepared/statistics_zeroshot_train.json ----
    zeroshot_target = prepared_root / "statistics_zeroshot_train.json"
    if not zeroshot_target.exists():
        if not target_stats.exists():
            raise SystemExit(f"missing {target_stats}; cannot build zeroshot stats")
        stats = json.load(open(target_stats, encoding="utf-8"))
        values = collect_categorical_values(parsed_root)
        changes = {}
        for name in CATEGORICAL:
            entry = stats.get(name)
            if entry is None or entry.get("type") != "categorical":
                raise SystemExit(f"feature {name!r} is not categorical in statistics: {entry}")
            vd = entry.setdefault("value_dict", {})
            missing = sorted(set(values[name]) - set(map(str, vd.keys())))
            next_index = max(vd.values()) + 1 if vd else 0
            for value in missing:
                vd[value] = next_index
                next_index += 1
            entry["no_vals"] = len(vd)
            changes[name] = missing
        with zeroshot_target.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, separators=(",", ":"))
        report["steps"].append(
            {
                "action": "derive",
                "source": str(target_stats),
                "target": str(zeroshot_target),
                "note": "appended categorical values missing from official statistics",
                "added": {name: missing for name, missing in changes.items()},
            }
        )
        print(f"derived {zeroshot_target} (added {changes})")
    else:
        print(f"present: {zeroshot_target}")

    manifest = {
        "generator": "prepare_stats.py",
        "note": "compatibility files derived from official OSF artifacts; "
                "official source and official OSF files are never modified.",
        "steps": report["steps"],
    }
    with (prepared_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"manifest written to {prepared_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
