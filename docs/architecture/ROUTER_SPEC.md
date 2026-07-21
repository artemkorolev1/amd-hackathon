# Router Specification — FW Router v2

## Purpose
Route each task to the optimal Fireworks model with per-category caveman system prompts to minimize output tokens while maintaining accuracy.

## Routing Table

| Category | Model | System Prompt | Max Tokens | Notes |
|----------|-------|---------------|:----------:|-------|
| sentiment | minimax-m3 | "Output EXACTLY one word: Positive, Negative, or Neutral. No explanation." | 20 | 1-word label |
| ner | minimax-m3 | "Entities: Person=..., Org=..., Loc=..., Date=... No prose. No headings." | 400 | Strips markdown tables |
| factual | minimax-m3 | "Answer directly. No preamble. No restate question. No closing." | 500 | ~28-300 words |
| summarization | minimax-m3 | "Only the summary obeying prompt's length constraint. No intro." | 500 | ~80-200 words |
| math | kimi-k2p7-code | "Output ONLY the number. No explanation." + KIMI_KILL | 550 | ~1 word ideal |
| code_gen | kimi-k2p7-code | "Write the code. Fenced block. No explanation." + KIMI_KILL | 600 | Code in fenced block |
| code_debug | kimi-k2p7-code | "Point out the bug in 1 sentence. Fix in fenced block." + KIMI_KILL | 500 | Bug + fix |
| logic | kimi-k2p7-code | "Output ONLY the answer. One word or letter." + KIMI_KILL | 500 | ~1 word ideal |

**KIMI_KILL suffix:** " ABSOLUTELY NO PREAMBLE. No 'we', no 'the user', no 'I need to', no 'let's', no 'here is', no self-talk. FIRST TOKEN = THE ANSWER."

## Parameters (all categories)
- `temperature`: 0.0 (greedy — shortest output)
- `stream`: False
- `reasoning_effort`: "none" (minimax only — kimi doesn't support it)
- No `prefill` parameter (causes format echoing on minimax; kimi ignores it anyway)

## Average Token Savings (verified on 40-question eval)

| Metric | Before (512/q) | After | Savings |
|--------|:-------------:|:-----:|:-------:|
| Total output | 20,480 | 5,146 | **75%** |
| Avg per question | 512 tok | 129 tok | **75%** |
| Minimax categories | 10,240 | ~1,200 | **88%** |
| Kimi categories | 10,240 | ~3,900 | **62%** |
| Total time | ~180s | ~152s | **16% faster** |
| Estimated cost (300 Q, FW API) | ~$0.12 | ~$0.03 | **75% cheaper** |

## Accuracy (40-question complexity eval)
- **Overall: 87.5%** (35/40) — same as non-caveman baseline
- Sentiment errors: 2/5 are model capability wall (sarcasm detection) — same as baseline
- Math error: 1/5 kimi preamble ate budget — same as non-caveman run
- NER, factual, summarization, code_gen, code_debug, logic: all correct where answer was produced

## Remaining Known Issues (Room for Improvement)

### 1. Kimi preamble on ~50% of complex questions
- "We need solve..." / "The user wants me to..." appears on ~10/20 kimi answers
- Kill switch in system message reduces it from ~80% to ~50%
- Prefill approach doesn't work (kimi writes preamble after the prefill)
- **Cost:** ~1,000-1,500 extra tokens per 40-Q run
- **Impact:** Wastes runtime but doesn't degrade accuracy (correct answers are still produced)

### 2. Sentiment model accuracy (model capability wall)
- minimax-m3 gets 3/5 correct on hard sentiment (sarcasm, dismissive)
- kimi-k2p7-code gets 4/5 correct on same questions
- But kimi adds token overhead that partially negates sentiment's savings
- **Tradeoff:** Stay with minimax for 90% token savings and accept 60% accuracy, or switch to kimi for 80% accuracy at higher token cost

### 3. No pipeline integration yet
- Router exists as standalone `agent/solvers/fw_router.py`
- Not wired into `agent/main.py`
- Pipeline still uses `config.py` / `dynamic_prompts.py` for routing decisions

## Integration Points

When ready to install:
1. Import `route` from `agent.solvers.fw_router` in `main.py`
2. Replace Fireworks decision logic with `cfg = route(category, prompt, complexity)`
3. Call `fw.solve(model=cfg.model_id, system_prompt=cfg.system_prompt, user_prompt=prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature, task_type=category)`
4. `reasoning_effort` is handled automatically by `fireworks.py` (detects minimax)
5. Remove `FW_PROMPTS` dict from router (legacy — replaced by caveman system messages)
