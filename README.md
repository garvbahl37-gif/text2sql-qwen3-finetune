# Fine-tuning Qwen3-4B for text-to-SQL

A 4-billion-parameter Qwen3 model fine-tuned to turn an English question and a
SQLite schema into a query that runs — trained on a laptop, measured against the
base model by executing every query against a real database, and served behind a
web demo.

| | |
|---|---|
| Demo | https://text2sql-qwen3.vercel.app |
| Model | https://huggingface.co/bharatverse11/qwen3-4b-text2sql |
| GGUF (CPU) | https://huggingface.co/bharatverse11/qwen3-4b-text2sql-gguf |

## Result

Held-out set of 300 examples the model never saw, greedy decoding, scored by
**execution accuracy**: build the database from each example's own schema, run
both the generated and the reference query, compare the result sets.

| metric | base Qwen3-4B | fine-tuned | change |
|---|---|---|---|
| **execution accuracy** | 78.3% | **84.0%** | **+5.7** |
| valid SQL rate | 97.7% | 98.3% | +0.7 |
| exact string match | 31.3% | 39.7% | +8.3 |

29 wins, 223 both right, 12 regressions, 36 both wrong.

The gains land where the base model was weak, and the dilution is visible: more
than half the evaluation set is "basic SQL" the base already handled.

| query type | n | base | fine-tuned |
|---|---|---|---|
| window functions | 10 | 30.0% | **60.0%** |
| single join | 38 | 63.2% | **79.0%** |
| aggregation | 71 | 70.4% | **80.3%** |
| basic SQL | 165 | 88.5% | 90.3% |
| subqueries | 13 | 69.2% | 61.5% |

Training cost 45 minutes on one Apple M5 Pro. No rented GPU.

## Whether the evaluation can be trusted

Execution accuracy is the headline because string matching undercounts: a
correct query written differently still counts here. Beyond that:

- **21% of the raw dataset has reference SQL that does not execute.** Every
  example is run before use and the broken ones are dropped, rather than
  training the model on invalid queries or scoring against them.
- Both models get an identical prompt and greedy decoding.
- The SQL extractor is deliberately lenient about markdown fences and preamble.
  The base model wraps answers in prose, and penalising formatting rather than
  correctness would inflate the result.
- Scoring lives in one module shared by both training backends, so a CUDA run
  and an Apple Silicon run cannot report subtly different numbers.

## End-to-end test

`tests/frontend_e2e.py` sends 18 questions across three schemas to the same API
the browser calls, executes the returned SQL, and compares against hand-written
reference queries.

```
executed without error : 18/18
matched the reference  : 16/18
needed a repair pass   : 0
latency                : 1.0-3.0s
```

Both misses are the metric rather than the model: asked which product sold the
most units it returned `('Widget', 416)` where the reference is `('Widget',)`.
The answer is right; strict result-set comparison penalises the extra column.
By content it scored 18 of 18.

All 18 reference queries were themselves verified to execute and return rows, so
a failure implicates the model and not the test.

## What did not work

Kept here because a portfolio of only successes is not evidence of much.

**Rebalancing the training mix by measured headroom.** More than half of run 1's
training data was "basic SQL" where the base model already scored 88.5%, while
window functions — 70 points of headroom — got 3%. Run 2 rebalanced toward the
hard classes, same size, same held-out set.

| | run 1, natural mix | run 2, balanced |
|---|---|---|
| execution accuracy | **84.0%** | 82.7% |

It did not help. **McNemar's test on the 300 paired examples gives p = 0.454** —
the runs disagree on 16 examples, ten one way and six the other, which is noise
inside one standard error. The hypothesis is unsupported and run 1 remains the
published model. `training/compare_runs.py` prints this comparison and refuses
to imply one when the runs are not actually comparable.

## Known weaknesses

- For hard analytical questions the model writes elaborate window-function
  queries and sometimes trips a SQLite rule, such as using a window function in
  `HAVING`. The serving layer retries once with the database's own error fed
  back, which fixes many of these. That mitigates a symptom; it does not fix the
  model.
- Greedy decoding reproduces the same wrong query however the repair prompt is
  worded — narrow supervised fine-tuning cost the model some
  instruction-following — so repairs sample instead.
- The model sometimes answers only part of a multi-clause question.

## Layout

```
training/   data preparation, training on two backends, evaluation, publishing
serving/    the inference server, deployable as a Hugging Face Space or locally
web/        Next.js frontend: playground, benchmark, method
tests/      end-to-end accuracy check against a deployed URL
```

## Reproduce

### Apple Silicon (free, ~45 min)

Unsloth needs CUDA, so the Apple path uses MLX instead.

```bash
cd training
set -a && . ../.env && set +a        # HF_TOKEN and HF_REPO
bash mac_pipeline.sh
```

Prepares data, trains, evaluates, fuses into full-precision weights and
publishes. Every stage is idempotent.

### CUDA — Kaggle (free) or RunPod (~$0.35)

```bash
cd training
export HF_TOKEN=hf_... HF_REPO=<user>/qwen3-4b-text2sql
bash runpod_bootstrap.sh
```

For Kaggle, open `training/kaggle_text2sql.ipynb`. Set the accelerator to
**GPU T4 x2** in the sidebar: Kaggle defaults to a P100, whose compute
capability 6.0 has no kernels in modern PyTorch builds, and the GPU type cannot
be set through the API. The notebook checks this and stops with instructions
rather than failing deep into the run.

### Configuration worth knowing

Every number in `mlx_lora_config.yaml` was measured on the target machine.

| config | throughput | peak memory | |
|---|---|---|---|
| batch 8, no checkpointing | — | OOM | |
| batch 4, checkpointing on | 1.4 ex/s | 6.0 GB | 3x slower for memory we have |
| batch 4, checkpointing off | 1.9 ex/s | 17.7 GB | swaps |
| **batch 2, checkpointing off** | **3.2 ex/s** | **9.1 GB** | chosen |

Three things that are easy to get wrong:

- **Sequence length.** The data's median example is 194 tokens and its longest
  is 581. The usual 2048 default is roughly three times oversized.
- **What an iteration is.** In MLX one iteration consumes one *batch*, and
  `grad_accumulation_steps` only changes how often the optimizer updates.
- **The learning-rate horizon.** The schedule advances once per optimizer
  *update*, so its decay window is `iters / grad_accumulation_steps`. Set it to
  `iters` and the rate barely decays at all.

`mac_pipeline.sh` checks all three against each other and refuses to start if
they disagree.

## Serving

`serving/app.py` runs either backend, chosen by environment variable:

| variable | effect |
|---|---|
| `USE_GGUF=0` | transformers on a GPU, full-precision weights |
| `USE_GGUF=1` | llama.cpp on CPU, q8_0 GGUF (6.7s/query at 2 threads) |
| `GGUF_PATH` | load a local `.gguf` and skip the Hub download |

It executes only read queries, against a throwaway in-memory database, and
refuses `DROP`, `ALTER`, `PRAGMA` and `ATTACH`.

**A note on free hosting.** A free Hugging Face account cannot host a
CPU-basic Gradio Space — that needs PRO — and a ZeroGPU Space will not run a
CPU-only app, since it expects GPU work to schedule. ZeroGPU itself works well
but has a small daily quota. The options that actually stay free are a local
server behind a Cloudflare tunnel, or a provider with free credits.

## Frontend

```bash
cd web && npm install
cp .env.example .env.local     # set HF_SPACE_URL, or MOCK_BACKEND=1 to work offline
npm run dev
```

The browser never talks to the model host directly; requests go through
`app/api/generate/route.ts`, so no token reaches the client.

## Security

- `.env` is gitignored and holds the Hugging Face token.
- The query executor allows reads only, on a disposable in-memory database.
- The API route validates and length-caps input before forwarding it.
