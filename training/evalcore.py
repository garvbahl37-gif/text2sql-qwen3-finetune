"""Scoring shared by both evaluation backends (CUDA/Unsloth and MLX/Apple).

Keeping extraction, execution scoring and report building in one place means the
two backends can never drift into reporting subtly different numbers.
"""
from __future__ import annotations

import re
from collections import defaultdict

from sqlutil import build_db, normalize_sql, results_match, run_sql

FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
SQL_KEYWORD = r"(?:WITH|SELECT|INSERT|UPDATE|DELETE)"
LINE_START = re.compile(rf"^\s*{SQL_KEYWORD}\b", re.IGNORECASE | re.MULTILINE)
ANYWHERE = re.compile(rf"\b{SQL_KEYWORD}\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull a query out of a raw completion.

    Deliberately generous: the base model wraps SQL in markdown and prose, and
    penalising it for formatting instead of correctness would inflate the
    fine-tuned model's apparent win. Returns "" when there is no query at all
    (e.g. the model refused), so it scores as invalid rather than as garbage.
    """
    if m := FENCE.search(text):
        text = m.group(1)
    if m := LINE_START.search(text):
        text = text[m.start() :]
    else:
        matches = list(ANYWHERE.finditer(text))
        if not matches:
            return ""
        text = text[matches[-1].start() :]
    return text.split(";")[0].strip()


def score_one(context: str, gold_sql: str, pred_sql: str) -> tuple[bool, bool]:
    """Return (executes_cleanly, result_set_matches_gold)."""
    if not pred_sql:
        return False, False
    try:
        con = build_db(context)
    except Exception:
        return False, False
    try:
        pred_rows = run_sql(con, pred_sql)
    except Exception:
        con.close()
        return False, False
    try:
        gold_rows = run_sql(con, gold_sql)
    except Exception:
        con.close()
        return True, False
    ok = results_match(gold_rows, pred_rows, gold_sql)
    con.close()
    return True, ok


def summarise(rows: list[dict], key: str) -> dict:
    n = max(len(rows), 1)
    return {
        "execution_accuracy": round(sum(r[f"{key}_exec_ok"] for r in rows) / n, 4),
        "valid_sql_rate": round(sum(r[f"{key}_runs"] for r in rows) / n, 4),
        "exact_match": round(sum(r[f"{key}_exact"] for r in rows) / n, 4),
    }


def build_report(examples: list[dict], base_raw: list[str], tuned_raw: list[str], **meta) -> tuple[dict, list[dict]]:
    """Execute every prediction and assemble the report the frontend renders."""
    rows = []
    for ex, b_raw, t_raw in zip(examples, base_raw, tuned_raw):
        b_sql, t_sql = extract_sql(b_raw), extract_sql(t_raw)
        b_runs, b_ok = score_one(ex["sql_context"], ex["gold_sql"], b_sql)
        t_runs, t_ok = score_one(ex["sql_context"], ex["gold_sql"], t_sql)
        gold_norm = normalize_sql(ex["gold_sql"])
        rows.append({
            "id": ex.get("id"),
            "domain": ex.get("domain", ""),
            "complexity": ex.get("complexity", ""),
            "question": ex["sql_prompt"],
            "schema": ex["sql_context"],
            "gold_sql": ex["gold_sql"],
            "base_sql": b_sql, "base_runs": b_runs, "base_exec_ok": b_ok,
            "base_exact": normalize_sql(b_sql) == gold_norm,
            "tuned_sql": t_sql, "tuned_runs": t_runs, "tuned_exec_ok": t_ok,
            "tuned_exact": normalize_sql(t_sql) == gold_norm,
        })

    metrics = {"base": summarise(rows, "base"), "tuned": summarise(rows, "tuned")}

    buckets = defaultdict(list)
    for r in rows:
        buckets[r["complexity"] or "unknown"].append(r)
    by_complexity = {
        name: {
            "n": len(b),
            "base": round(sum(r["base_exec_ok"] for r in b) / len(b), 4),
            "tuned": round(sum(r["tuned_exec_ok"] for r in b) / len(b), 4),
        }
        for name, b in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    }

    wins = [r for r in rows if r["tuned_exec_ok"] and not r["base_exec_ok"]]
    both = [r for r in rows if r["tuned_exec_ok"] and r["base_exec_ok"]]
    losses = [r for r in rows if not r["tuned_exec_ok"] and r["base_exec_ok"]]
    neither = [r for r in rows if not r["tuned_exec_ok"] and not r["base_exec_ok"]]

    report = {
        **meta,
        "dataset": "gretelai/synthetic_text_to_sql",
        "n_examples": len(rows),
        "decoding": "greedy",
        "metrics": metrics,
        "deltas": {k: round(metrics["tuned"][k] - metrics["base"][k], 4) for k in metrics["base"]},
        "by_complexity": by_complexity,
        "counts": {
            "tuned_win": len(wins), "both_correct": len(both),
            "tuned_regression": len(losses), "both_wrong": len(neither),
        },
        # Mostly wins, but keep honest failures in so the page isn't a highlight reel.
        "samples": wins[:14] + both[:4] + losses[:3] + neither[:3],
    }
    return report, rows


def print_table(metrics: dict, counts: dict) -> None:
    print("\n" + "=" * 62)
    print(f"{'metric':<22}{'base':>12}{'fine-tuned':>14}{'delta':>12}")
    print("-" * 62)
    for k in metrics["base"]:
        b, t = metrics["base"][k], metrics["tuned"][k]
        print(f"{k:<22}{b:>11.1%}{t:>13.1%}{t - b:>+11.1%}")
    print("=" * 62)
    print(f"wins {counts['tuned_win']}  |  both right {counts['both_correct']}  "
          f"|  regressions {counts['tuned_regression']}  |  both wrong {counts['both_wrong']}")
