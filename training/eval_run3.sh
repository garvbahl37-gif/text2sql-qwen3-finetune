#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
A=outputs/qwen3-4b-text2sql-run3
echo "[eval] started $(date '+%H:%M:%S')"
$PY evaluate_mlx.py --adapter "$A" --test data/multilevel_test.jsonl --limit 150 \
    --out outputs/run3-multilevel/eval_report.json
echo "[eval] MULTILEVEL_DONE $(date '+%H:%M:%S')"
$PY evaluate_mlx.py --adapter "$A" --test data/test.jsonl --limit 300 \
    --out outputs/run3-general/eval_report.json --skip-base
echo "[eval] GENERAL_DONE $(date '+%H:%M:%S')"
