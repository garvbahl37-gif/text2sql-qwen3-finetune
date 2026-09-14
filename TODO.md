# Status

Updated after run 2 and the Vercel deployment.

## Live

| thing | where | notes |
|---|---|---|
| Frontend | https://text2sql-qwen3.vercel.app | public, no auth wall |
| Model weights | https://huggingface.co/bharatverse11/qwen3-4b-text2sql | 8.04 GB bf16 |
| GGUF (CPU) | https://huggingface.co/bharatverse11/qwen3-4b-text2sql-gguf | 4.28 GB q8_0 |
| Inference Space | https://huggingface.co/spaces/bharatverse11/text2sql-qwen3 | llama.cpp on CPU |
| Code | https://github.com/garvbahl37-gif/text2sql-qwen3-finetune | public |

## Results so far

Scored on 300 held-out examples, identical across runs, by execution accuracy:
build the database from each example's schema, run both the generated and the
reference query, compare result sets.

| run | training data | execution accuracy |
|---|---|---|
| base Qwen3-4B | none | 78.3% |
| **run 1** | 4,000, natural mix | **84.0%** |
| run 2 | 4,000, headroom-balanced | 82.7% |

**Run 2 did not improve on run 1.** McNemar on the paired examples gives
p = 0.454 — they disagree on 16 of 300, ten one way and six the other. The
difference is noise, inside one standard error (+/-2.2 points). Rebalancing the
mix by measured headroom is an unsupported hypothesis, not a win. Run 1 is the
published model.

## Done

- [x] Data pipeline that drops the ~21% of examples whose reference SQL does not execute
- [x] QLoRA fine-tune on Apple Silicon (MLX) and CUDA (Unsloth), one recipe, two backends
- [x] Execution-accuracy evaluation, shared scoring so backends cannot disagree
- [x] Merge to full precision and publish to the Hub
- [x] Frontend: playground, benchmark with real numbers, method
- [x] Serving Space with a self-repair pass (feeds the SQLite error back)
- [x] Second dataset (`b-mc2/sql-create-context`, Spider/WikiSQL derived)
- [x] Headroom-balanced training mix — tried, measured, did not help
- [x] Deploy to Vercel
- [x] q8_0 GGUF for CPU inference (6.7s/query at 2 threads, measured)
- [x] 18-case end-to-end test harness with hand-verified reference queries

## End-to-end test through the deployed frontend

18 questions across three schemas, sent to the same `/api/generate` the browser
uses, each compared against a hand-written reference query.

| | result |
|---|---|
| executed without error | **18/18** |
| matched the reference exactly | **16/18 (89%)** |
| needed a repair pass | 0 |
| latency | 1.0-3.0s |

By type: basic 7/7, join 4/4, filter 1/1, multi-join 1/1, aggregation 3/4,
subquery 0/1.

Both misses are the metric, not the model. Asked which product sold the most
units it answered `('Widget', 416)` where the reference is `('Widget',)`, and
for departments above the average cost it answered `('cardiology', 470.25)`
against `('cardiology',)`. The answers are right; strict result-set comparison
penalises the extra column. By content the model scored 18/18.

## Resume here

Run 3 was stopped part-way at iteration 125 of 2700. A checkpoint exists at
iteration 250, so it does not restart from zero.

```bash
cd training
# data is already prepared: 5,400 examples = 4,000 natural gretel mix
# + 1,400 synthetic multi-level aggregation (26%)
../.venv/bin/python -m mlx_lm lora -c mlx_lora_config.yaml \
    --resume-adapter-file outputs/qwen3-4b-text2sql-run3/0000250_adapters.safetensors
```

Then both evaluations, which is the point of the run:

```bash
# 1. the original held-out 300, byte-identical to run 1 -- checks for regression
../.venv/bin/python evaluate_mlx.py --adapter outputs/qwen3-4b-text2sql-run3 \
    --test data/test.jsonl --limit 300 --out outputs/run3-general/eval_report.json --skip-base
# 2. 150 held-out multi-level examples, zero overlap with training -- measures the fix
../.venv/bin/python evaluate_mlx.py --adapter outputs/qwen3-4b-text2sql-run3 \
    --test data/multilevel_test.jsonl --limit 150 --out outputs/run3-multilevel/eval_report.json
../.venv/bin/python compare_runs.py outputs/run1-natural-mix/eval_report.json \
    outputs/run3-general/eval_report.json --labels "run1,run3"
```

Notes for tomorrow:

- Training ran at 0.44 it/s rather than run 1's 0.74, because the local demo
  backend was holding the 4.3GB GGUF and swap was at 7.4GB. Stop that server
  first and it should run closer to 0.74, so roughly an hour rather than two.
- If run 3 does not beat run 1 on the general set, run 1 stays published. Run 2
  already demonstrated that a plausible data change can fail to help.
- The demo currently points at the Hugging Face Space. A Cloudflare quick tunnel
  gets a fresh random hostname on every start, so for a stable local backend use
  Tailscale Funnel, an ngrok static domain, or a Cloudflare named tunnel.

## Open

- [ ] Restore a permanent backend. The demo currently runs through a Cloudflare
      quick tunnel to a local llama.cpp server, so it only works while that
      machine is awake, and the URL changes on restart.
- [ ] Kaggle 20k two-dataset run — needs Accelerator set to **GPU T4 x2** in the
      notebook UI. Kaggle defaults to P100, which modern PyTorch has no kernels
      for, and the GPU type cannot be set through the API.
- [ ] Quantify how much the self-repair pass actually adds to execution accuracy
- [ ] Decide whether to publish a run-2 model at all (currently: no, it is not better)

## Known weaknesses

- The model writes elaborate window-function queries for hard analytical
  questions and sometimes trips SQLite rules. The serving layer retries once
  with the error fed back; that mitigates the symptom, it does not fix the model.
- Greedy decoding reproduces the same wrong query however the prompt is worded,
  so repairs sample instead. Narrow SFT cost the model some instruction-following.
- Repairs are non-deterministic, and the model sometimes drops part of a
  multi-clause question.

## Costs

Everything is on free tiers. Training ran locally on an M5 Pro (~45 min per run),
weights and Space are on Hugging Face free, frontend on Vercel free. A free HF
account cannot host a CPU-basic Gradio Space (that needs PRO), so the Space runs
llama.cpp on the CPU of a ZeroGPU Space, which uses no GPU quota.
