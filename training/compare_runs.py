"""Compare evaluation reports from several training runs side by side.

Only valid when every run was scored on the same held-out examples, so this
refuses to print a comparison it cannot vouch for.

    python compare_runs.py outputs/run1-natural-mix/eval_report.json outputs/eval_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ["execution_accuracy", "valid_sql_rate", "exact_match"]


def load(p: Path) -> dict:
    r = json.loads(p.read_text())
    if not r.get("metrics"):
        raise SystemExit(f"{p} has no metrics (placeholder?)")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--labels", help="comma separated names, one per report")
    args = ap.parse_args()

    reports = [load(p) for p in args.reports]
    labels = (args.labels.split(",") if args.labels
              else [p.parent.name or p.stem for p in args.reports])
    if len(labels) != len(reports):
        raise SystemExit("--labels count must match the number of reports")

    sizes = {r["n_examples"] for r in reports}
    if len(sizes) > 1:
        print(f"WARNING: runs scored on different example counts {sizes}; "
              "the comparison is not apples to apples.\n")

    base = reports[0]["metrics"]["base"]
    if any(r["metrics"]["base"] != base for r in reports):
        print("WARNING: the base model scored differently across runs, so the "
              "held-out set or decoding was not identical.\n")
    else:
        print(f"Base model identical across all runs (n={reports[0]['n_examples']}), "
              "so these are directly comparable.\n")

    w = max(len(l) for l in labels) + 2
    print(f"{'metric':<22}{'base':>9}" + "".join(f"{l:>{w}}" for l in labels))
    print("-" * (31 + w * len(labels)))
    for m in METRICS:
        row = f"{m:<22}{base[m]:>8.1%}"
        for r in reports:
            row += f"{r['metrics']['tuned'][m]:>{w}.1%}"
        print(row)

    print(f"\n{'vs base (points)':<22}{'':>9}" + "".join(f"{l:>{w}}" for l in labels))
    print("-" * (31 + w * len(labels)))
    for m in METRICS:
        row = f"{m:<22}{'':>8}"
        for r in reports:
            row += f"{(r['metrics']['tuned'][m] - base[m]) * 100:>+{w}.1f}"
        print(row)

    print(f"\n{'regressions':<22}{'':>9}" + "".join(f"{l:>{w}}" for l in labels))
    for key, name in (("tuned_win", "wins"), ("tuned_regression", "regressions")):
        row = f"{name:<22}{'':>8}"
        for r in reports:
            row += f"{r.get('counts', {}).get(key, 0):>{w}}"
        print(row)

    classes = sorted({c for r in reports for c in r.get("by_complexity", {})},
                     key=lambda c: -reports[0].get("by_complexity", {}).get(c, {}).get("n", 0))
    if classes:
        print(f"\nexecution accuracy by complexity")
        print(f"{'class':<20}{'n':>5}{'base':>8}" + "".join(f"{l:>{w}}" for l in labels))
        print("-" * (33 + w * len(labels)))
        for c in classes:
            ref = reports[0].get("by_complexity", {}).get(c, {})
            row = f"{c:<20}{ref.get('n', 0):>5}{ref.get('base', 0):>7.0%}"
            for r in reports:
                v = r.get("by_complexity", {}).get(c)
                row += f"{v['tuned']:>{w}.0%}" if v else f"{'-':>{w}}"
            print(row)

    for r, l in zip(reports, labels):
        n = r.get("train_examples")
        print(f"\n{l}: trained on {n if n else '?'} examples, backend {r.get('backend','?')}")


if __name__ == "__main__":
    main()
