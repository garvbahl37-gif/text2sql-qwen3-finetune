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

## Open

- [ ] Run the 18-case suite against the live Vercel URL once the Space finishes building
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
