# Fine-tuning Qwen3-4B for text-to-SQL

A complete, reproducible fine-tuning project: data preparation, LoRA training,
an honest base-vs-tuned benchmark, a free inference backend, and a frontend that
runs the generated SQL against a real database.

```
training/   data prep, training (Apple Silicon + CUDA), evaluation, publish
serving/    Hugging Face Space that serves the model (free)
web/        Next.js frontend for Vercel (free)
```

## What this demonstrates

The headline is not "a model was trained". It is **execution accuracy measured
against the base model on held-out data**: build the SQLite database from each
test example's own schema, run both the generated and the reference query, and
compare result sets. A correct query written differently still counts, and
examples whose reference query fails to run are dropped so neither model is
punished for a broken gold answer.

Both models get the identical prompt and greedy decoding, and the SQL extractor
is deliberately lenient about markdown fences and preamble — the base model
wraps answers in prose, and penalising formatting instead of correctness would
inflate the result. Scoring lives in one module (`evalcore.py`) shared by both
training backends, so they cannot drift into reporting different numbers.

Worth knowing: **21% of the raw dataset has gold SQL that does not execute.**
`prepare_data.py` builds a real database from every example's schema and drops
those, rather than training on broken queries.

---

## 1. Train

Three paths. They produce the same artifact and share every other script.

| | Apple Silicon (MLX) | Kaggle (CUDA + Unsloth) | RunPod (CUDA + Unsloth) |
|---|---|---|---|
| Cost | **free** | **free** (30 GPU-h/week) | ~$0.35 |
| Time | ~21 min (measured, M5 Pro) | not measured — watch the first 50 steps | ~45 min |
| Needs | M-series Mac, ~10GB free memory | Kaggle account, phone-verified | rented A40 / L40S |
| Ties up your laptop | yes | no | no |

### Apple Silicon

Unsloth cannot be used here — it requires CUDA (Triton kernels, bitsandbytes
4-bit), neither of which runs on Metal. MLX is Apple's equivalent and is what
`mlx_lora_config.yaml` targets.

```bash
cd training
set -a && . ../.env && set +a     # loads HF_TOKEN and HF_REPO
bash mac_pipeline.sh
```

That creates the venv, prepares data, trains, evaluates, fuses and publishes.
Every stage is idempotent, so a re-run skips work that already succeeded.

### Kaggle

Open `training/kaggle_text2sql.ipynb` on Kaggle. Set **Accelerator** to `GPU T4 x2`,
turn **Internet** on (needs phone verification), and add your Hugging Face write
token as a Secret named `HF_TOKEN` — the notebook reads it from Kaggle Secrets so
it never appears in a cell.

The notebook fetches this code either by cloning your GitHub repo (set `REPO_URL`)
or from the `training/` folder uploaded as a Kaggle Dataset, so the repo can stay
private.

A Kaggle **API key is not needed for the data** — the dataset comes from Hugging
Face, not Kaggle. A key only matters if you want to upload the code as a Dataset
or push the notebook from the CLI (`kaggle kernels push`).

Note that T4 is Turing and has no bf16; `train.py` detects that and selects fp16
automatically.

### RunPod

Rent one A40 / L40S / A6000 with the PyTorch 2.4+ template, 60GB container disk.

```bash
cd training
export HF_TOKEN=hf_...
export HF_REPO=bharatverse11/qwen3-4b-text2sql
bash runpod_bootstrap.sh
```

**Terminate the pod when it finishes.** RunPod bills per second for as long as
the pod exists, whether or not anything is running on it.

## Why the Apple Silicon config looks the way it does

Every number in `mlx_lora_config.yaml` was measured on the target machine
(M5 Pro, 24GB unified, 16-core GPU), not guessed.

| Config | Throughput | Peak memory | |
|---|---|---|---|
| batch 8, no checkpointing | — | OOM | |
| batch 4, checkpointing on | 1.4 ex/s | 6.0 GB | 3x slower for memory we have |
| batch 4, checkpointing off | 1.9 ex/s | 17.7 GB | swaps, so it loses to batch 2 |
| **batch 2, checkpointing off** | **3.2 ex/s** | **9.1 GB** | chosen |

Three things that are easy to get wrong here:

- **Sequence length.** The prepared data has a median of 194 tokens and a
  longest example of 581. The usual 2048 default is ~3x oversized; 640 fits
  every example whole and is most of the speedup.
- **What an iteration is.** In MLX one iter consumes one *batch*, and
  `grad_accumulation_steps` only changes how often the optimizer updates. So
  one epoch is `examples / batch_size` iters.
- **The LR schedule horizon.** It advances once per optimizer *update*, not per
  iteration, so its decay window is `iters / grad_accumulation_steps` — 250,
  not 2000. Set it to 2000 and the learning rate barely decays at all.

`mac_pipeline.sh` checks all three against each other and refuses to start if
they disagree, rather than letting you discover it an hour later.

### Knobs worth turning

| Want | Change |
|---|---|
| Better numbers | `TRAIN_SIZE=8000` (~42 min), then set `iters: 4000` and decay `500` |
| Faster first run | `TRAIN_SIZE=2000`, `iters: 1000`, decay `125` (~11 min) |
| Out of memory | drop `batch_size` to 1 and double `grad_accumulation_steps` |
| Other apps open | set `grad_checkpoint: true` — slower, but peaks at ~6GB |

## 2. Publish the eval report

```bash
cp training/outputs/eval_report.json web/data/eval_report.json
```

Until you do, the Benchmark tab says the numbers have not been measured —
nothing on the site is simulated.

## 3. Serve the model (free)

Create a Hugging Face Space, SDK **Gradio**, and upload the contents of
`serving/`. Set the Space variable `MODEL_ID=bharatverse11/qwen3-4b-text2sql`.

Hardware *ZeroGPU* for free H200 slices (fast, quota-limited), or *CPU basic*
with `USE_GGUF=1` for slow but unlimited inference. See `serving/README.md`.

## 4. Deploy the frontend (free)

```bash
cd web
vercel --prod
```

Set one environment variable in the Vercel project:

```
HF_SPACE_URL=https://bharatverse11-<space-name>.hf.space
```

The browser never talks to Hugging Face directly — requests go through
`app/api/generate/route.ts`, so no token is exposed client-side.

To work on the UI locally without running the model:

```bash
cd web && cp .env.example .env.local   # set MOCK_BACKEND=1
npm install && npm run dev
```

---

## Design notes

The frontend is styled as a query console rather than a landing page: hairline
rules, lowercase filename labels (`schema.sql`, `query.sql`), no shadows or
gradients, and IBM Plex throughout — SQL was designed at IBM in 1974, which
makes it the one type choice specific to this project.

In the benchmark, base versus fine-tuned is encoded as **outline versus solid**
rather than red versus green, so colour never has to mean "good" or "bad" and
the single accent stays available for interface signals.

## Security

- `.env` is gitignored and holds the HF token. **Rotate that token** — it was
  pasted into a chat session.
- The `KGAT_…` key is stored unlabelled in `.env`; it matches neither RunPod's
  `rpa_` nor Hugging Face's `hf_` format. Identify it or delete it.
- The Space executes only read queries, against a throwaway in-memory database,
  and `DROP` / `ALTER` / `PRAGMA` / `ATTACH` are refused outright.
- The Vercel route validates and length-caps input before forwarding it.
