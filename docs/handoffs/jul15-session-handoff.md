# Session Handoff — July 15, 2026

## Summary

This session focused on three things: (1) JAPA→GEPA rename across all files, (2) fixing the 8-way cascade classifier from 82.9% to 98.7% across 5 datasets, and (3) building a binary cascade decision tree for code tool routing with category-specific LLM prompts.

---

## Active State

### Classifier Cascade Accuracy (validated)

| Dataset | Questions | Stage 2 (primary) | Cascade (full) | New 3 secondaries |
|---------|:---------:|:-----------------:|:--------------:|:-----------------:|
| training-v2 | 1,514 | 83.2% | **98.2%** | Added |
| training-v3 | 152 | 82.9% | **98.7%** | Added |
| validation-v1 | 400 | — | **99.0%** | Added |
| validation-v2 | 400 | — | **96.2%** | Added |

### Tool Routing — Pipeline-Context (Deterministic solvers only)

| Category | Accuracy | Solver |
|----------|:-------:|--------|
| factual | 90-94% | FactDB |
| sentiment | 68-76% | VADER |
| ner | 80% | solve_ner (old regex) |
| code_debug | 10% | pattern solver |
| logic | 6-26% | logical_reasoning |
| math | 0-2% | LLM-only |
| code_gen | 0-6% | LLM-only (template solver on exact fn match only) |
| summarization | 0% | LLM-only |

### Binary Cascade Tree for Coding Tools (`agent/solvers/code_tool_cascade.py`)

```
code_gen prompt
  N0: Has structured I/O (JSON examples/test cases)?
  ├─YES→ N1: Tree/graph/linked list?  → prompt: coding_challenge_ds
  │       └─NO→ N2: DP problem?       → prompt: coding_challenge_dp
  │               └─NO→ N3: Sort/search/math? → prompt: coding_challenge_sort_search
  │                       └─NO→ llm_formal     → prompt: coding_challenge_formal
  └─NO→ N4: Function name matches template?  → template solver (exact name)
          └─NO→ N5: Known algorithm named?    → template solver (renamed)
                  └─NO→ llm_simple             → default code_gen prompt
```

### 4 new prompt sets in `dynamic_prompts.py`

| Prompt Key | For | When |
|-----------|-----|------|
| `coding_challenge_ds` | Tree/graph/linked list problems | N1=YES |
| `coding_challenge_dp` | Dynamic programming | N2=YES |
| `coding_challenge_sort_search` | Sorting/searching/math | N3=YES |
| `coding_challenge_formal` | Generic structured challenge | Otherwise formal |

---

## Decisions Made This Session

1. **JAPA→GEPA rename:** All 33+ files renamed + 2 Hermes skills. Zero JAPA remains.
2. **Cluster 7 fixes into classifier cascade:** harness.py import, reasoning_secondary zebra guard, competition regex, factual guard return, NER task guard, "who was" factual boost, "Law and Order" logic guard.
3. **3 new secondaries built:** secondary_qa.py, secondary_codeguard.py, secondary_nertweet.py.
4. **Evaluation methodology corrected:** Isolated solver eval with substring match is misleading — always test through pipeline with official fuzzy_match grader.
5. **Solver order fixed:** solve_ner (old, 80%) must be BEFORE solve_ner_v3 (prototype, 66%) in the chain.
6. **Routing config:** code_gen, math, summarization are genuinely LLM-only. No deterministic solver helps.
7. **Binary cascade for coding tools:** Each node is one yes/no classifier. The tree grows naturally — add binary splits as new categories emerge.
8. **Not wired into pipeline.py:** The code_tool_cascade exists and validates but `pipeline.process()` still uses the old `_run_deterministic()` flat loop.

---

## Files Changed

### New files
- `agent/solvers/code_tool_cascade.py` — Binary cascade tree for coding solvers
- `agent/solvers/ner_solver.py` — Purpose-built NER (not as good as old solve_ner)
- `agent/solvers/cascade_router.py` — DEPRECATED, was renamed from this concept
- `scripts/eval/train_ner_solver.py` — NER training eval
- `scripts/eval/eval_pipeline_solvers.py` — Pipeline-context solver eval
- `scripts/eval/eval_tool_routing.py` — Tool routing eval
- `scripts/eval/val_cascade_router.py` — Cascade router validation
- `scripts/eval/val_code_tool_cascade.py` — Code tool cascade validation
- `agent/secondary_qa.py` — Factual↔summarization disambiguator
- `agent/secondary_codeguard.py` — Code_gen guard for MCQ formatting
- `agent/secondary_nertweet.py` — NER↔factual for tweets/biomedical

### Modified files
- `agent/classifier.py` — Wired 3 new secondaries
- `agent/category_filter.py` — Competition regex fix, factual guard fix, NER guard, "who was" boost
- `agent/secondary_reasoning.py` — Zebra puzzle guard for "Solve:" prefix
- `agent/secondary_summarization.py` — SQuAD guard + MCQ choices guard in CASE 4
- `agent/dynamic_prompts.py` — Added 4 coding_challenge_* prompt sets
- `agent/pipeline.py` — Solver order fixed (ner before ner_v3)
- `scripts/harness.py` — Expanded from 4 to 8 categories in DET_CAT_MAP, added missing imports
- `scripts/eval/eval_classifiers.py` — Fixed path resolution
- `PROJECT_LOG.md` — Updated JAPA→GEPA
- All analysis files renamed: japa_*.md → gepa_*.md

---

## Blockers

1. ~~code_tool_cascade not wired into pipeline.py. The `process()` method still uses `_run_deterministic()` flat loop. Need to plug `route_code()` into the code_gen/code_debug path.~~ **FIXED — wired in Jul 15 session. Cascade intercepts code_gen/code_debug, returns template solver hits directly, and sets coding_challenge_* prompt keys for LLM paths.**
2. **Full pipeline eval with GPU not run.** All classifier numbers are deterministic-only. Need to run with local LLM (gemma-3-1b / qwen) to measure end-to-end answer accuracy per category.
3. ~~CONTEXT.md is stale. Shows v0-v5 data from July 9-10. The dataset analysis, classifier improvements, and cascade tree are not reflected.~~ **Updated Jul 15**

---

## Next Session Priorities

1. ~~Wire code_tool_cascade into pipeline.py's process() method~~ **DONE**
2. Run full pipeline eval with GPU (gemma-3-1b for code, qwen2.5-1.5b for rest)
3. ~~Evaluate binary cascade tree routing decisions on Kaggle dataset (616 problems)~~ **DONE — all 616 have structured I/O, 201 DS/98 DP/112 sort/205 generic**
4. Consider building out the tree with more binary splits as new pattern categories emerge from eval
