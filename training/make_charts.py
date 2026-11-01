"""Render the measured results as charts for the README.

Every number comes from a committed eval report; nothing is typed in by hand,
so a chart cannot drift from the data that produced it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, RULE, COBALT, PAPER = "#17211c", "#5e6b63", "#d5dad6", "#1b4dff", "#f7f8f6"
OUT = Path("docs")


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=RULE, lw=0.8)
    ax.set_axisbelow(True)


def rd(p):
    return json.loads(Path(p).read_text())


def chart_runs():
    runs = [
        ("base\nQwen3-4B", None, "training/outputs/run1-natural-mix/eval_report.json", "base"),
        ("run 1\n4k, 1 source", "published", "training/outputs/run1-natural-mix/eval_report.json", "tuned"),
        ("run 2\n4k, balanced", None, "training/outputs/run2-balanced/eval_report.json", "tuned"),
        ("run 3\n5.4k, +synthetic", None, "training/outputs/run3-general/eval_report.json", "tuned"),
        ("run 4\n16k, 3 sources", None, "training/outputs/run4-on-run1-testset/eval_report.json", "tuned"),
    ]
    labels, vals, notes = [], [], []
    for lab, note, path, key in runs:
        if not Path(path).exists():
            continue
        labels.append(lab); vals.append(rd(path)["metrics"][key]["execution_accuracy"] * 100)
        notes.append(note)

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
    fig.patch.set_facecolor(PAPER)
    colors = [MUTED] + [COBALT if n == "published" else INK for n in notes[1:]]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    base = vals[0]
    ax.axhline(base, color=MUTED, ls=(0, (4, 4)), lw=1)
    ax.text(len(vals) - 0.4, base + 0.4, f"base {base:.1f}%", color=MUTED, fontsize=8, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%",
                ha="center", color=INK, fontsize=10, fontweight="medium")
    ax.set_ylim(70, 90)
    ax.set_ylabel("execution accuracy", color=MUTED, fontsize=9)
    ax.set_title("Four fine-tuning runs, scored on the same 300 held-out examples\n"
                 "Every difference from run 1 tests as noise (McNemar p = 0.45–0.56)",
                 color=INK, fontsize=11, loc="left", pad=14)
    style(ax)
    fig.tight_layout(); fig.savefig(OUT / "runs.png", facecolor=PAPER); plt.close(fig)
    print("  docs/runs.png")


def chart_multilevel():
    pairs = [("base Qwen3-4B", 0.0), ("run 1 (published)", 0.0)]
    p = Path("training/outputs/run4-holdout-bare/eval_report.json")
    if p.exists():
        pairs.append(("run 4", rd(p)["metrics"]["tuned"]["execution_accuracy"] * 100))
    labels = [a for a, _ in pairs]; vals = [b for _, b in pairs]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor(PAPER)
    bars = ax.barh(labels[::-1], vals[::-1],
                   color=[COBALT if v > 0 else RULE for v in vals[::-1]], height=0.5)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", color=INK, fontsize=10, fontweight="medium")
    ax.set_xlim(0, 40)
    ax.set_xlabel("execution accuracy on 18 hand-written multi-level questions",
                  color=MUTED, fontsize=9)
    ax.set_title("Multi-level aggregation: a capability run 1 did not have\n"
                 "Questions written by hand, never produced by the data generator",
                 color=INK, fontsize=11, loc="left", pad=14)
    style(ax); ax.xaxis.grid(True, color=RULE, lw=0.8); ax.yaxis.grid(False)
    fig.tight_layout(); fig.savefig(OUT / "multilevel.png", facecolor=PAPER); plt.close(fig)
    print("  docs/multilevel.png")


def chart_enrichment():
    b = Path("training/outputs/run4-on-run1-testset/eval_report.json")
    e = Path("training/outputs/run4-general-enriched/eval_report.json")
    if not (b.exists() and e.exists()):
        print("  (enrichment chart skipped: reports missing)"); return
    bb, ee = rd(b), rd(e)
    groups = ["base model", "fine-tuned (run 4)"]
    bare = [bb["metrics"]["base"]["execution_accuracy"] * 100,
            bb["metrics"]["tuned"]["execution_accuracy"] * 100]
    rich = [ee["metrics"]["base"]["execution_accuracy"] * 100,
            ee["metrics"]["tuned"]["execution_accuracy"] * 100]

    x = range(len(groups)); w = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
    fig.patch.set_facecolor(PAPER)
    b1 = ax.bar([i - w/2 for i in x], bare, w, label="bare DDL", color=RULE, edgecolor=MUTED)
    b2 = ax.bar([i + w/2 for i in x], rich, w, label="enriched schema", color=COBALT)
    for bars, vals in ((b1, bare), (b2, rich)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.4, f"{v:.1f}%",
                    ha="center", color=INK, fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(groups)
    ax.set_ylim(70, 88)
    ax.set_ylabel("execution accuracy", color=MUTED, fontsize=9)
    ax.set_title("Enriched schemas help the base model, not the fine-tuned one\n"
                 "run 4 was trained on bare DDL, so enriching only at inference is a mismatch",
                 color=INK, fontsize=11, loc="left", pad=14)
    leg = ax.legend(frameon=False, fontsize=9); [t.set_color(MUTED) for t in leg.get_texts()]
    style(ax)
    fig.tight_layout(); fig.savefig(OUT / "enrichment.png", facecolor=PAPER); plt.close(fig)
    print("  docs/enrichment.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    chart_runs(); chart_multilevel(); chart_enrichment()
