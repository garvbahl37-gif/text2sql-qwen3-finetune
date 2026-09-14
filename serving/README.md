---
title: Qwen3-4B Text-to-SQL
emoji: 🗃️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Qwen3-4B fine-tuned to write SQLite queries, executed live
---

# Qwen3-4B Text-to-SQL

Inference backend for a Qwen3-4B-Instruct model LoRA fine-tuned on
[`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql).

Given a SQLite schema and a natural-language question it generates a query,
**executes it against a throwaway in-memory database**, and returns the rows.

## Space variables

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_ID` | `unsloth/Qwen3-4B-Instruct-2507` | merged model repo to serve |
| `USE_GGUF` | `0` | `1` switches to llama.cpp CPU inference |
| `GGUF_REPO` | `${MODEL_ID}-gguf` | GGUF repo used when `USE_GGUF=1` |
| `MAX_NEW_TOKENS` | `320` | generation cap |

## Two ways to run this Space

**ZeroGPU (default, fast).** Hardware → *ZeroGPU*. Free H200 slices, subject to a
per-account quota. Keep `requirements.txt` as-is.

**Free CPU tier (slow, unlimited).** Hardware → *CPU basic*. Replace
`requirements.txt` with `requirements-cpu.txt` and set `USE_GGUF=1`. Roughly
5-8 tok/s — a SQL query takes about 6-10 seconds.

You need a q4_k_m GGUF for this mode. On CUDA, `merge_and_push.py --gguf`
produces one. On Apple Silicon it does not: `mlx_lm.fuse --export-gguf` supports
only llama/mixtral/mistral architectures, not Qwen3. Convert the published
weights with llama.cpp instead:

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py <your-merged-model-dir> --outtype f16 --outfile qwen3-4b-text2sql-f16.gguf
./llama-quantize qwen3-4b-text2sql-f16.gguf qwen3-4b-text2sql-q4_k_m.gguf Q4_K_M
huggingface-cli upload <user>/qwen3-4b-text2sql-gguf qwen3-4b-text2sql-q4_k_m.gguf
```

## HTTP API

The frontend calls this Space rather than embedding a model. Gradio's REST API
is a two-step call:

```bash
EVENT=$(curl -s -X POST https://<user>-<space>.hf.space/gradio_api/call/generate \
  -H 'Content-Type: application/json' \
  -d '{"data":["CREATE TABLE t(a INT);","how many rows?",320,0]}' | jq -r .event_id)

curl -N https://<user>-<space>.hf.space/gradio_api/call/generate/$EVENT
```

The completion event carries `[sql, result_markdown, metadata]`.
