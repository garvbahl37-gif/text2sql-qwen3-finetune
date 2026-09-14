# Fine-tuning Qwen3-4B for text-to-SQL

A complete, reproducible fine-tuning project: data preparation, QLoRA training,
an honest base-vs-tuned benchmark, a free inference backend, and a frontend that
runs the generated SQL against a real database.

```
training/   data prep, QLoRA training, evaluation, merge + publish
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
inflate the result.

## Cost

| Stage | Where | Cost |
|---|---|---|
| Training (~45 min on one 48GB GPU) | RunPod A40 / L40S | ~$0.35 |
| Weights hosting | Hugging Face Hub | free |
| Inference | Hugging Face Space (ZeroGPU, or CPU + GGUF) | free |
| Frontend | Vercel | free |

To make it **entirely free**, train on Kaggle instead (30 GPU-hours/week on
T4×2). Same scripts; expect roughly 3 hours rather than 45 minutes.

---

## 1. Train

Rent one A40 / L40S / A6000 on RunPod with the PyTorch 2.4+ template, 60GB
container disk. Then:

```bash
git clone <this repo> && cd FIneTuning/training

export HF_TOKEN=hf_...                        # write scope
export HF_REPO=<your-hf-username>/qwen3-4b-text2sql

bash runpod_bootstrap.sh
```

That installs dependencies, prepares data, trains, evaluates, merges, and
publishes. Every stage is idempotent, so a re-run skips work that already
succeeded. Or run the steps yourself:

```bash
python prepare_data.py --train-size 8000 --test-size 300
python train.py --data data --out outputs/qwen3-4b-text2sql-lora
python evaluate.py --adapter outputs/qwen3-4b-text2sql-lora --limit 300
python merge_and_push.py --adapter outputs/qwen3-4b-text2sql-lora --repo $HF_REPO --gguf
```

**Terminate the pod when it finishes.** RunPod bills per second for as long as
the pod exists, whether or not anything is running on it.

### Why it trains this fast

- 4-bit base weights with LoRA adapters, so only ~1% of parameters get gradients
- Unsloth's fused kernels, roughly 2× faster than stock PEFT at lower VRAM
- Loss masked to the assistant turn, so the model is graded on the SQL it writes
  rather than on re-predicting the schema it was handed
- 8,000 examples, one epoch — a narrow task does not need more

### Knobs worth turning

| Want | Change |
|---|---|
| Better numbers | `--train-size 20000 --epochs 2` (~2.5h) |
| Cheaper / faster | `--train-size 4000 --rank 16` (~20 min) |
| Bigger model | `--model unsloth/Qwen3-8B --batch-size 4 --grad-accum 4` |
| Out of memory | halve `--batch-size`, double `--grad-accum` |

## 2. Publish the eval report

Copy `training/outputs/eval_report.json` to `web/data/eval_report.json`. Until
you do, the Benchmark tab says the numbers have not been measured — nothing on
the site is simulated.

## 3. Serve the model (free)

Create a Hugging Face Space, SDK **Gradio**, and upload the contents of
`serving/`. Set the Space variable `MODEL_ID` to your merged repo.

**ZeroGPU** (default, fast): set Hardware to *ZeroGPU*. Free H200 slices,
subject to a per-account quota.

**Free CPU tier** (slow, unlimited): set Hardware to *CPU basic*, replace
`requirements.txt` with `requirements-cpu.txt`, and set `USE_GGUF=1`. Needs the
GGUF export from `merge_and_push.py --gguf`. About 5-8 tokens/sec, so a query
takes 6-10 seconds.

## 4. Deploy the frontend (free)

```bash
cd web
vercel --prod
```

Set one environment variable in the Vercel project:

```
HF_SPACE_URL=https://<your-username>-<space-name>.hf.space
```

The browser never talks to Hugging Face directly — requests go through
`app/api/generate/route.ts`, so no token is exposed client-side.

To work on the UI locally without a GPU:

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

- `.env` is gitignored. The key pasted during setup is stored there unlabelled
  and **should be rotated** — it was exposed in chat.
- The Space executes only read queries, against a throwaway in-memory database,
  and `DROP` / `ALTER` / `PRAGMA` / `ATTACH` are refused outright.
- The Vercel route validates and length-caps input before forwarding it.
