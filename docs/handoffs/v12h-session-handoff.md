# Handoff — Session v12h: Fireworks Model Router Design

## Current state

**Git:** v12d branch on filtered-build, with uncommitted v12h changes:
- `agent/solvers/fw_router.py` — **standalone** Fireworks model router (importable, self-testing)
- `agent/solvers/fireworks.py` — Fireworks API solver (stdlib urllib, no deps)
- `agent/dynamic_prompts.py` — MAX_TOKENS all set to 200 (universal budget)
- `run_v12h.py` — hybrid runner (Phi-4 local + FW fallback) for the 300-set
- `test_fw_prompts.py` — prompt style tests for available FW models
- `~/.fireworks_key` — API key saved

## Fireworks available models on this account

| Model ID | Type | Best for |
|:---------|:-----|:---------|
| `accounts/fireworks/models/gpt-oss-120b` | General 120B | sentiment, code, factual, summarization |
| `accounts/fireworks/models/deepseek-v4-pro` | Reasoning | math, logic, NER |
| `accounts/fireworks/models/kimi-k2p6` | General | fallback (has prompt leakage issues) |

## 300-set run results — what we learned

### Model performance by category

| Category | Local Phi-4 | FW (kimi-k2p7-code) | Best model (proven) |
|:---------|:-----------:|:-------------------:|:-------------------:|
| sentiment | 42.9% | **87.5%** | **gpt-oss-120b** (one-word system msg) |
| math | **55%** | 67% (small sample) | **deepseek-v4-pro** (terse, outputs "3") |
| code_gen | 100% | 50% (prompt leakage) | **gpt-oss-120b** (system msg→actual code) |
| code_debug | 100% | — | gpt-oss-120b (same pattern as code_gen) |
| ner | 67% (det solver) | 100% (3/3 fallbacks) | deepseek-v4-pro or local |
| factual | 87% | 100% (4/4) | local or gpt-oss-120b |
| logic | 94% | 100% (6/6) | deepseek-v4-pro or local |
| summarization | 90% | **50%** | **revert to local Phi-4** |
| general | 100% | — | local |

### Key findings

1. **kimi-k2p7-code (old) is bad** — prompt leakage on code/summarization, echoes instructions back
2. **gpt-oss-120b is the best general-purpose FW model** — no prompt leakage with system messages
3. **deepseek-v4-pro is excellent for math** — outputs terse correct answers like "3" with no preamble
4. **deepseek still has prompt leakage for NER/code** — needs `[TASK]` prefix trick to suppress
5. **Summarization is worse on FW** — local Phi-4 at 90% beats every FW model tried
6. **Sentiment significantly improved** — 42.9% → 87.5% with FW; gpt-oss can do one-word answers

### Routing rules baked into fw_router.py

```
sentiment     → gpt-oss-120b   (30 tok, system msg → one word)
math          → deepseek-v4-pro (300 tok, user msg direct)
code_gen      → gpt-oss-120b   (300 tok, system msg)
code_debug    → gpt-oss-120b   (250 tok, system msg)
ner           → deepseek-v4-pro (300 tok, [TASK] prefix)
factual       → gpt-oss-120b   (200 tok, system msg)
logic         → deepseek-v4-pro (250 tok, user msg direct)
summarization → gpt-oss-120b   (200 tok, system msg)
general       → gpt-oss-120b   (200 tok, system msg)
```

## What the router component is

`agent/solvers/fw_router.py` — a **standalone, importable module** with zero pipeline dependencies:

- `route(category, prompt, complexity)` → returns `RouterConfig(model_id, prompt_prefix, max_tokens, temperature, label)`
- Prompts are short custom prefixes (NOT the local-LLM system prompts that cause leakage)
- Self-test mode: `python3 agent/solvers/fw_router.py`

## What needs to happen next (next session)

### 1. Integrate fw_router into the pipeline
- In `run_v12h.py` or a new v12i runner: import `agent.solvers.fw_router`
- Replace hardcoded `FIREWORKS_CATEGORIES` / `NAKED_CATEGORIES` with router-based decisions
- For categories where router returns a FW model → call FW directly
- For categories where router says local → use local Phi-4 with max_tokens=200

### 2. Fix NER deterministic solver collapse
- Remove NER from `DETERMINISTIC_CATEGORIES` in the runner
- Local Phi-4 scored 86% on NER (NAKED) — just let it use the LLM
- Or route to FW via router (deepseek with `[TASK]` prefix)

### 3. Fix summarization misclassification
- A pre-check was added in `run_v12h.py`: if `len(prompt) > 1000` and `"SOURCE" in prompt`, force to summarization
- This needs to be ported to the integrated pipeline
- After fixing, summarization should go back to local Phi-4 (90% accuracy)

### 4. Run full 300-set eval with the integrated router
- FIREWORKS_API_KEY is saved at `~/.fireworks_key`
- Run on GPU first, then simulate CPU timing
- Compare accuracy vs the 83.0% hybrid baseline

### Known issues / things to test
- **deepseek NER prompt** — `[TASK]` prefix trick tested for code but not confirmed for NER yet
- **gpt-oss code gen** — tested with Kadane (works), but needs broader validation across all code questions
- **Complexity scaling** — the complexity multiplier in `get_max_tokens` applies on top of router's max_tokens. Decide whether to keep or disable.
- **max_tokens=200 universal** — factual collapsed from 100% to 12.5% at 120 tok, recovered to 87% at 200 tok. 200 seems safe for all categories now.
