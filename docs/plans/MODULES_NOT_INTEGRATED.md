# Modules Not Integrated Into Main Pipeline

> Generated: 2026-07-13
> Method: Actual file-by-file import tracing of all .py files in agent/, staging/, runner/, and root.
> Three entry points checked: (1) dispatcher→harness→agent.Pipeline, (2) dispatcher→staging/entrypoint, (3) runner/*
> Constraint: agent/ is UNTOUCHABLE; staging/ and runner/ are modifiable.
> Excludes: .venv/, __pycache__/, .git/, node_modules/, tests/, container/, scripts/, gepa_plans/, archive/

---

## How to read this

A module is **"Integrated"** if at least one of the three production code paths imports it (directly or transitively).  
**"Not Integrated"** means zero production code paths touch it.

| Bucket | Meaning |
|--------|---------|
| **A** | Active & Integrated — in a production path |
| **B** | Standalone — built, works, has tests, but nobody imports it from a production path |
| **C** | Orphaned / Dead — not imported by anything at all, or only by other orphaned files |

This report lists **only Buckets B and C**.

---

## Summary

| Scope | Files | Bucket B | Bucket C |
|-------|-------|----------|----------|
| agent/ (core) | 33 .py files | 6 | 11 |
| agent/solvers/ | 17 .py files | 8 | 5 |
| staging/ | 11 .py files | 0 | 0 |
| runner/ | 6 .py files | 0 | 0 |
| root | 3 .py files | 0 | 1 |
| **Total** | **~70** | **14** | **17** |

---

## BUCKET B: Standalone — Built But Not Wired

These modules exist, work, and could be wired in, but no production path imports them.

### agent/main.py — Full Filtered Pipeline (v9)

- **Why not integrated:** Not imported by any entry point. harness.py imports `agent.Pipeline` from `agent.pipeline.py`, not `main.py`. The `main.py` file is a separate, self-contained pipeline implementation with its own flow (T0/T1 → S2 → S3 → S4 → solvers → QC gate).
- **What it does:** 444-line async pipeline with stage0, stage2 (category_filter.classify), stage3 (complexity_filter.score), stage4 (decision table), deterministic solvers, Fireworks escalation, local consensus, code quality retry loop, and circuit breaker.
- **Modules only it uses (would go dead if main.py removed):** `agent.quality_config`, `agent.circuit_breaker`
- **To wire it in:** Replace `from agent import Pipeline` in `harness.py` with `from agent.main import main` and call it; or make `driver.py` delegate to `agent.main` when an env var is set.

### agent/classifier.py — Full Classification Pipeline

- **Why not integrated:** Not imported by any production module. Agent pipeline imports `agent.category_filter` directly — bypassing `classifier.py` entirely. The classifier module exists as a convenience wrapper (8-way primary + 3 secondary disambiguators + NER solver chain).
- **What it does:** Chains `category_filter.classify_with_detail` (primary 8-way scorer) with `secondary_code.resolve_code`, `secondary_factual.resolve_factual`, and `secondary_reasoning.resolve_reasoning`. Also exposes `classify_ner()` which calls `prototype_ner_v3` with fallback to `deterministic.solve_ner`.
- **Only wired into:** `agent/ml_classifier.py` does NOT import it. `agent/hierarchical_classifier.py` duplicates its logic (and is also orphaned). Standalone only.
- **To wire it in:** Use `classifier.classify()` instead of `category_filter.classify_with_detail()` in `pipeline.py`'s routing, or import it from `staging/entrypoint.py` as a richer classification option.

### agent/ml_classifier.py — ML-based Classifier (sklearn)

- **Why not integrated:** Not imported by any production module. Uses `sklearn` pipeline (TfidfVectorizer + LogisticRegression) for category classification as an alternative to the pure-regex `category_filter.py`. Only useful if ML classification is preferred over heuristic.
- **What it does:** Loads a trained sklearn model, uses `category_filter` via importlib as fallback. 172 lines.
- **To wire it in:** Import into `pipeline.py` or `staging/entrypoint.py` as a primary classifier with regex fallback.

### agent/summarization.py — Fireworks Summarization Router

- **Why not integrated:** Not imported by any production module. Reroutes headline-generation tasks to Fireworks API because 1B models score 0% on xsum. 196 lines.
- **What it does:** Detects headline-format prompts, calls Fireworks API for summarization. Could complement `solve_summarization` in the deterministic solver path.
- **To wire it in:** Add an import in `pipeline.py`'s summarization handling path, or register in `tool_registry`.

### agent/quality_config.py — QC Configuration Data

- **Why not integrated:** Only imported by `agent/main.py` (also not integrated). Contains per-category QC thresholds (top3_gap, margin, inverse_active). Zero code, pure data (60 lines).
- **What it does:** Structured QC config dictionary with per-category thresholds tuned for 85.4% classifier accuracy.
- **To wire it in:** Import it into the pipeline's QC gate path (`pipeline.py`'s verify step or `agent/solvers/verify.py` itself).

### agent/circuit_breaker.py — Fireworks Circuit Breaker

- **Why not integrated:** Only imported by `agent/main.py`. The production `pipeline.py` does NOT implement circuit breaker logic — it simply catches exceptions and continues. The staging judge also doesn't use it.
- **What it does:** Tracks consecutive non-retryable 4xx errors, opens circuit for a cooldown period, supports half-open probes. 175 lines, stdlib only.
- **To wire it in:** Add to `agent/pipeline.py`'s Fireworks escalation path, and/or to `staging/ready_judge.py`'s escalation path.

### agent/solvers/local.py — Local llama.cpp Server Solver

- **Why not integrated:** Not imported by pipeline.py (which uses `local_vote.py` instead). Only imported by standalone scripts: `run_v12e.py`, `smoke_test.py`, `_test_local_model.py`. Uses urllib to talk to llama.cpp server (not the direct Python binding).
- **What it does:** Communicates with llama.cpp REST API, multi-round tool execution, deadline-based escalation.
- **To wire it in:** Replace or complement `local_vote.py` in `pipeline.py` with this if using server-mode llama.cpp instead of direct Python binding.

### agent/solvers/{logic_solver, code_sandbox, spell_check, web_search, easter_egg_shelf} — 5 Tool Solvers

- **Why not integrated:** Only imported by:
  - `agent/solvers/__init__.py` (NOT integrated — no production path imports it)
  - `agent/solvers/tool_registry.py` (lazy imports, NOT integrated)
- **What they do:**
  - `logic_solver` — Constraint-based puzzle solver (python-constraint, 384 lines)
  - `code_sandbox` — Safe Python execution (RestrictedPython, 216 lines)
  - `spell_check` — SymSpell spelling correction (140 lines)
  - `web_search` — DuckDuckGo search (131 lines)
  - `easter_egg_shelf` — Fun utilities (CSV format, text stats, word games, 378 lines)
- **To wire them in:** Three options: (1) Import `agent/solvers/__init__.py` from pipeline, (2) Import individual tools into pipeline directly, or (3) Wire `tool_registry.py` into the production path so its 28 registered tools become callable.

### agent/solvers/tool_registry.py — 28-Tool Registry

- **Why not integrated:** Only imported by `eval_tools.py` and `text_processor.py` (both orphaned). Contains the full inventory of 28 deterministic tools (math_solve, sentiment_analysis, ner_extract, factual_qa, logical_reasoning, code_debug, web_search, spell_check, etc.) with all their metadata.
- **What it does:** 367-line framework with `@tool` decorator pattern (inspired by smolagents), lazy-imports all solver modules. Would be the single integration point for ALL deterministic tools.
- **To wire it in:** Import into pipeline.py and use `tool_registry.get_tool(name)` as the solver dispatch mechanism instead of the current hardcoded import list.

### agent/solvers/logic_reasoning.py — LSAT Logical Reasoning Solver

- **Why not integrated:** Only lazy-imported by `tool_registry.py` (also not integrated). Not in any production path.
- **What it does:** Heuristic LSAT-style argument analysis: strengthen, weaken, assumption, inference, flaw, main_point, explain questions. 582 lines, stdlib only.
- **To wire it in:** Import into pipeline.py's logic solver path, or via tool_registry integration.

### agent/solvers/deterministic_filters.py — Pre-filters for Tool Routing

- **Why not integrated:** Only imported by `eval_tools.py` (orphaned). Contains per-category pre-filters (math, code, sentiment, etc.) that check if a prompt is suitable for a deterministic solver before calling it. 248 lines.
- **What it does:** Regex-based filters that prevent wasting time on prompts a tool can't handle.
- **To wire it in:** Import into deterministic solver dispatch in pipeline.py.

### agent/solvers/eval_tools.py — Tool Evaluation Harness

- **Why not integrated:** Standalone eval script only. Runs all registered deterministic tools against labeled eval datasets and produces accuracy reports. 491 lines.
- **What it does:** Maps categories to tools, loads eval sets, runs tools, compares outputs to expected answers, reports accuracy. Imports `tool_registry` and `deterministic_filters`.
- **To wire it in:** This is an eval tool, not meant for production. Keep standalone or promote to `runner/evaluate.py` integration.

---

## BUCKET C: Orphaned / Dead

These modules are not imported by anything (or only by other orphaned modules). They could be archived.

### agent/caveman_prompts.py — Caveman System Prompts

- **Why dead:** Superseded by `agent/solvers/fw_router.py` (which has identical caveman prompts embedded in `FORMAT_PROMPTS`) and `agent/dynamic_prompts.py` (which has more sophisticated tiered prompts). Not imported by any module.
- **Action:** Archive.

### agent/bitmorphic_classifier.py — 7-Signal Complexity Scorer

- **Why dead:** Has known bugs. Superseded by `agent/complexity.py` (MiniLM-L6-v2 + LogisticRegression) and `agent/complexity_filter.py` (per-category heuristic). Only imported by `agent/category_router.py` (also dead).
- **Action:** Archive.

### agent/category_router.py — Deterministic Task Router

- **Why dead:** Only imports `bitmorphic_classifier` (dead). Not imported by any module. Falls back to 'other_complex'. 407 lines. The routing logic is now done inline in `pipeline.py` and `agent/solvers/fw_router.py`.
- **Action:** Archive.

### agent/hierarchical_classifier.py — Secondary Classifier Orchestrator

- **Why dead:** Not imported by any module. Superseded by `agent/classifier.py`'s inline importlib approach. Duplicates the secondary classifier logic (imports `secondary_code`, `secondary_factual`, `secondary_reasoning`). 89 lines.
- **Action:** Archive.

### agent/run_logger.py — Per-Query Pipeline Logger

- **Why dead:** Not imported by any production path. Only used by standalone `multi_runner.py` (root and scripts/). 471 lines, writes detailed pipeline traces to .xlsx. Could be useful for debugging but isn't wired.
- **Action:** Move to `scripts/` or wire into pipeline.py's process() method.

### agent/{gepa_runner.py, generate_generation_0.py, eval_gen0_qwen.py} — GEPA Modules

- **Why dead:** GEPA genetic algorithm experiment modules. `gepa_runner.py` (908 lines) is the main GA optimizer. The other two are helpers. Only import each other, nothing else imports them. Not production code.
- **Action:** Keep in experimental/ or move to `gepa_plans/`.

### agent/{secondary_code.py, secondary_factual.py, secondary_reasoning.py} — Secondary Classifiers

- **Why dead:** Only imported by:
  - `agent/classifier.py` (via importlib) — which is **Bucket B** (standalone, not wired)
  - `agent/hierarchical_classifier.py` (via direct import) — which is **Bucket C** (dead)
- **What they do:** Pure-regex disambiguators for known 8-way category confusions (code_debug↔code_gen, logic↔math, factual↔logic/math). Total: ~1,373 lines of well-tested deterministic logic.
- **To wire them in:** Wire `agent/classifier.py` (Bucket B) into the production path, or import them directly into `pipeline.py`'s classification stage.
- **Note:** These are high-quality deterministic modules with zero external deps. Unused working code.

### agent/solvers/text_processor.py — Text Preprocessing Module

- **Why dead:** Not imported by any module at all. Contains smart chunker, JSON validator, markdown cleaner, etc. Imports `tool_registry` (also dead). 586 lines.
- **Action:** Archive, or move to scripts/.

### agent/solvers/{prototype_ner_v2.py, prototype_ner_solver.py} — NER Prototypes

- **Why dead:** Superseded by `prototype_ner_v3.py`. Not imported by anything. `prototype_ner_v3.py` is also not integrated (Bucket B — imported by standalone classifier.py only).
- **Action:** Archive.

### agent/solvers/{prototype_zebra_solver.py, prototype_zebra_v2.py} — Zebra Puzzle Solvers

- **Why dead:** Not imported by anything. Prototype constraint-puzzle solvers for logic puzzles (441 + 155 lines). The production pipeline handles logic puzzles via `deterministic.solve_logic` or Fireworks escalation.
- **Action:** Archive or move to gepa_plans/.

### agent/solvers/upgrade_deterministic.py — Deterministic Upgrade Script

- **Why dead:** Script that reads deterministic.py and produces improvements. Build/dev tool, not runtime code. 1,178 lines.
- **Action:** Archive.

### root/multi_runner.py — Multi-Model Runner

- **Why dead:** Standalone script that runs one eval set through multiple GGUF models sequentially. Imports agent modules directly (pre_filter, category_filter, complexity, deterministic, dynamic_prompts, run_logger). Not imported by any path.
- **Action:** Move to scripts/ or keep as standalone debugging tool.

---

## Key Findings

1. **The secondary classifiers are good code going to waste.** `secondary_code.py`, `secondary_factual.py`, and `secondary_reasoning.py` (~1,373 lines total, pure stdlib, no external deps) are proven to fix 227+ classification errors on the 300-set. They're only disconnected because `classifier.py` (their only caller) is standalone. Wiring `classifier.py` into the pipeline would immediately improve category accuracy.

2. **tool_registry.py is the missing integration layer.** All 28 deterministic tools are already registered, documented, and lazy-loadable. If `tool_registry.py` were imported by `pipeline.py`, every solver module (logic_solver, code_sandbox, spell_check, web_search, easter_egg_shelf, fact_db, logic_reasoning) would become reachable through a single import.

3. **quality_config.py and circuit_breaker.py are valuable but stranded.** They're only used by `main.py` (a parallel pipeline implementation). The production `pipeline.py` has no QC config or circuit breaker — these would improve robustness if wired in.

4. **solvers/__init__.py is a dead import hub.** Nobody imports it from a production path. It eagerly imports logic_solver, code_sandbox, easter_egg_shelf, spell_check, web_search — but these modules are also dead because the init file isn't in any import chain.

5. **~31 modules (44% of agent/) are not in any production path.** The active agent/ codebase is roughly 18 modules for the pipeline + 6 GEPA modules. Everything else is disconnected.

---

## Quick-Win Integration Priorities

| Priority | Module(s) | Effort | Impact |
|----------|-----------|--------|--------|
| P1 | Wire `classifier.py` → production | 1 import change | Fixes 227+ classification errors via secondary disambiguators |
| P2 | Wire `tool_registry.py` → production | 1 import change | Makes 28 tools available through one integration point |
| P3 | Add `quality_config.py` to pipeline's QC gate | 1 import + minor refactor | Better QC threshold tuning |
| P4 | Add `circuit_breaker.py` to pipeline's Fireworks path | 1 import + minor refactor | Graceful degradation on 4xx errors |
| P5 | Connect `secondary_*.py` directly to pipeline's classify step | Small refactor | Same as P1 but more surgical |
