#!/usr/bin/env bash
# Run 3: run 1's natural mix plus 1,400 synthetic multi-level aggregation
# examples. Evaluates twice -- on the original held-out set to catch regression,
# and on held-out multi-level examples to measure the fix.
set -uo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
A=outputs/qwen3-4b-text2sql-run3

# MLX truncates over-long sequences with only a warning. Truncating a gold SQL
# answer mid-query teaches the model to emit unfinished statements, so check first.
CAP=$(awk '/^max_seq_length:/{print $2}' mlx_lora_config.yaml)
$PY check_lengths.py --data data/train.jsonl --max-seq "$CAP" || exit 1

echo "[run3] training started $(date '+%H:%M:%S')"
$PY -m mlx_lm lora -c mlx_lora_config.yaml
echo "[run3] TRAINING_DONE $(date '+%H:%M:%S')"

$PY evaluate_mlx.py --adapter "$A" --test data/test.jsonl --limit 300 \
    --out outputs/run3-general/eval_report.json --skip-base
echo "[run3] GENERAL_EVAL_DONE $(date '+%H:%M:%S')"

$PY evaluate_mlx.py --adapter "$A" --test data/multilevel_test.jsonl --limit 150 \
    --out outputs/run3-multilevel/eval_report.json
echo "[run3] MULTILEVEL_EVAL_DONE $(date '+%H:%M:%S')"
