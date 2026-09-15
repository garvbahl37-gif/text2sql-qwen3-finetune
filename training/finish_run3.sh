#!/usr/bin/env bash
# Finish run 3 from the iteration-2250 checkpoint: the last 450 iterations,
# with the learning-rate schedule continued rather than restarted.
set -uo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
A=outputs/qwen3-4b-text2sql-run3

$PY check_lengths.py --data data/train.jsonl --max-seq 640 || exit 1

echo "[finish] resuming from iter 2250 at $(date '+%H:%M:%S')"
$PY -m mlx_lm lora -c mlx_lora_config_resume.yaml \
    --resume-adapter-file /tmp/resume_from_2250.safetensors
echo "[finish] TRAINING_DONE $(date '+%H:%M:%S')"

$PY evaluate_mlx.py --adapter "$A" --test data/multilevel_test.jsonl --limit 150 \
    --out outputs/run3-multilevel/eval_report.json
echo "[finish] MULTILEVEL_DONE $(date '+%H:%M:%S')"

$PY evaluate_mlx.py --adapter "$A" --test data/test.jsonl --limit 300 \
    --out outputs/run3-general/eval_report.json --skip-base
echo "[finish] GENERAL_DONE $(date '+%H:%M:%S')"
