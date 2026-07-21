# Handoff: v12c → v12d

## What v12c is

Locked submission image. Full pipeline (Stage 0→2→3→deterministic→Fireworks→local). Sentiment/ner/math/logic always route to Fireworks. Everything else uses complexity >= 0.2 threshold.

**Docker:** `ghcr.io/artemkorolev1/amd-hackathon-submit:v12c` (sha256:5fba8abb82e0)
**Git tag:** `v12c` on filtered-build worktree (commit 5726a06)
**Submit dir:** `/home/artem/dev/amd-hackathon-submit/` — LOCKED, contains VERSION_LOCKED.md

## Key files

| File | Location |
|------|----------|
| Pipeline entry | `/home/artem/dev/amd-hackathon-submit/agent/main.py` |
| Stage 2 classifier | `/home/artem/dev/amd-hackathon-submit/agent/stage2.py` |
| Config (FW thresholds) | `/home/artem/dev/amd-hackathon-submit/agent/config.py` |
| Dynamic prompts | `/home/artem/dev/amd-hackathon-submit/agent/dynamic_prompts.py` |
| Working worktree | `/home/artem/dev/amd-hackathon-filtered-build/` |

## Eval data (all results from tonight)

All in `/home/artem/dev/amd-hackathon-filtered-build/eval_results/`:
- `eval_comprehensive_20260711_203219.json` — latest 300-set eval with full sensor matrix (76.0%, 228/300)
- `eval_comprehensive_20260711_190653.json` — previous 300-set eval (76.0%, same accuracy — the update was config changes, not model changes)
- `fireworks_test_results.json` — Fireworks vs local on 17 failure cases: **Fireworks fixed 16/17 (94%)**
- `EVAL_REPORT_v11_300_QWEN25.md` — full v11 report
- `EVAL_REPORT_v9_GPU_20260711.md` — earlier 60-set report

Analysis tools in `eval_results/`:
- `generate_sensor_matrix.py` — produces the full sensor matrix report
- `analyze_fireworks_markers.py` — analyzes which signals predict model failures
- `test_fireworks.py` — runs Fireworks against failure cases

## Eval dataset

`/home/artem/dev/amd-hackathon-shared/eval_all_300.json` — 300 self-generated questions, used for all local evals. Not the grader's actual seed bank (expected 5-10pp gap to grader score).

## What was learned

1. **4 categories hit the 1.5B's capability wall**: sentiment (100% model limit), ner (100%), math (94%), logic (77%). All others: failures are classifier errors or grader fuzzy_match issues, not model capability.
2. **Fireworks fixes 94% of failure cases** (16/17 tested).
3. **Stage 3 complexity scorer is poor** — can't distinguish easy from hard. Most questions score < 0.3.
4. **QC gate is counter-predictive** — don't use it for Fireworks escalation.
5. **Router (19-cat) is 57.3%** — Stage 2 (85.0%) is the right classifier.

## v12d plan: Swap to Nemotron

**Goal:** Replace the local model with Nemotron-3-Nano-4B, keep everything else identical.

**Model file:** `/home/artem/dev/amd-hackathon-parallel/models/nemotron-3-nano-4b-Q4_K_M.gguf` (2.7 GB)

**Changes needed in submit dir:**
1. `agent/config.py` — change `LOCAL_MODEL_PATH` or rely on `MODEL_PATH` env var override
2. `agent/solvers/local_vote.py` — model path resolution: Nemotron may need different n_ctx or prompt format
3. Dockerfile — change the curl download URL to Nemotron GGUF, or COPY the local file
4. Dockerfile — line 34: `curl -L ... /models/qwen2.5-1.5b-instruct-q4_k_m.gguf` → Nemotron URL

**Nemotron prompt format:** Nemotron uses Llama 3 chat format (`<|begin_of_text|><|start_header_id|>system<|end_header_id|>...`), NOT ChatML. Check the `local_vote.py` chat_completion function — it may need format adjustment.

Keep:
- Stage 2/3, dynamic prompts, Fireworks thresholds, QC — unchanged
- No parallelization (match v12c's behavior)
- All config from v12c

## Nemotron baseline scores (from earlier session)

From the Jul 11 baseline eval:
- Nemotron-3-Nano-4B: **90.7% on 300 questions** (vs Qwen2.5-1.5B: 79.7%)
- Dominates math (96%), NER (100%), factual (94%)
- 2.7 GB vs 1.1 GB — but same pipeline should accept it

**Expected improvement:** The 90.7% was on bare questions with no pipeline. With our full pipeline (Stage 2 routing, dynamic prompts, deterministic solvers, Fireworks escalation), the score should be higher. The 1.5B model is the bottleneck for 4 categories — Nemotron should shrink that gap significantly.
