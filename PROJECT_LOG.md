# AMD Hackathon — Project Log

> Auto-generated project log. Updated daily by cron job.
> For permanent reference knowledge, see AIVAULT wiki.

## Daily Log

### 2026-07-13 — Post-Hackathon Retrospective & Infrastructure Build

- Conducted full retrospective covering 18 Hermes sessions across 8 days
- Analyzed root causes: CPU overload from parallel agents, worktree explosion (7 dirs, 41GB), scope creep past v6.1 (84.2% → never submitted v12d at 93.7%)
- Patched 4 Hermes skills: multi-agent-dispatcher (role definitions + TLDR + model routing), context-md-logging (verbosity levels), ce-worktree (max 3 worktrees, auto-commit), agentic-engineering (eval save protocol)
- Created dispatcher profile at ~/.hermes/profiles/dispatcher/ with auto-load of dispatcher skills
- Pinned sub-agent model to deepseek-v4-flash with auto-approve enabled
- Created cron jobs: daily project log update (20:00 CDT), weekly cleanup (Monday 06:00 CDT)
- Created worktree auto-commit hook at ~/.hermes/scripts/worktree-auto-commit.sh

### 2026-07-14 — GEPA Analysis, Summarization Solver & Factual Ablation

- Built summarization solver with eval datasets (train/val/hard_test) and ran initial GEPA evolution — metric mismatch identified (fuzzy_match unsuitable for summarization/logic)
- Ran GEPA (Genetic Pareto Algorithm) analysis across all 8 categories: NER identified as format-ceiling at 36.8% with local 1B models, factual knowledge capped at ~84% by FactDB gaps rather than prompt strategy
- Conducted factual ablation experiments — analyzed 77 questions, confirmed FactDB FTS5 as the real lever (21K facts), recommended BM25 tuning + verification reranking
- Created per-category architecture references and multi-step worker research documents for NER, local-only and per-category routing strategies

### 2026-07-15 — Full GEPA Prompt Evolution & Dynamic Prompts Integration

- Completed GEPA prompt evolution on all 8 categories with local GGUF models (qwen2.5-1.5b/coder-1.5b): reached 100% on sentiment and code_debug, 60% code_gen, 50% math, 36.8% NER (format ceiling)
- Updated fw_router.py with format-prompt routing for NER (routes to prototype_ner_v3 instead of local LLM) and integrated prototype_ner_v3 into main pipeline
- Wrote GEPA-optimized prompts into agent/dynamic_prompts.py "low" complexity tier for sentiment, code_debug, code_gen, math
- Created comprehensive GEPA results summary documenting bottlenecks, per-category best prompts, and recommended pipeline updates

### 2026-07-15 — Code Reorganization, Cascade Solver Pipeline & Staging Updates

- Reorganized root-level scripts into structured subdirectories under scripts/ (analysis/, benchmarks/, eval/, tools/) for maintainability
- Added cascade solvers for all 8 categories (sentiment, code_debug, code_gen, NER, math, logic, summarization, factual) with cascade routing, format normalization, and tool-based dispatch
- Wired cascade solvers into the agent pipeline with cell runner, workflow gate, resource manager, contracts, and GEPA pipeline integration
- Updated staging worker pool, entrypoint, judge, and build scripts with training data v3 pipeline and GPU eval tools
- Created comprehensive documentation: system architecture diagrams, handoff docs, GEPA analysis reports, colab notebook, and reference architecture docs

### 2026-07-16/17 — DeepSeek Multi-Agent Eval, Pipeline RCA Fixes & Template Math Solver

- Explored DeepSeek-R1-Distill-Qwen-1.5B across all 8 categories — Qwen2.5-1.5B pipeline remains leader at ~75% overall vs DeepSeek ~55%, though DeepSeek wins on math (+12pp), sentiment (+12pp), and logic (+8pp) raw
- Root-caused pipeline scoring ~55% on 300-set: fixed 6 bugs including hardcoded 0.0 temperature, routing table not passing temperature, and incorrect reasoning_keywords — score improved +7pp to 62.3%
- Fixed critical executor bug in `agent/pipeline.py` that was silently killing all LLM inference — recovered ~+20pp overall; built template-based math solver (+31.6pp, reaching 100% training math)
- Expanded FactDB from 21K to 112K facts by integrating NQ-Open (87,925 train + 3,610 dev); synced 46.8MB DB to main repo for Docker builds
- Fixed summarization pipeline with two-step entity extraction strategy (+10.5pp), fixed 3 remaining math failures (math 84.2% → 100%), and hardened logic deterministic solvers (propositional chains, water jug BFS, truth-teller fixes)

### 2026-07-21 — v12e Math Pipeline Expansion, ToRA Enhancements & Finalization

- **v12e math template expansion**: 5 new templates (rate_work, buy_sell, pct_of_total, group_sharing, pop_growth) → 11 total template types
- **Qwen2.5-Coder for math extraction**: switched to coder model for better JSON-structured parameter extraction from word problems
- **Anti-hallucination T11 prompts**: type A/B prompt variants with validation guards
- **Temperature tuning**: extraction at temp=0.3 balanced accuracy vs regressions on edge cases
- **ToRA solver enhancements**: variable extraction pre-filter for multi-variable problems; consensus voting across 4 temperatures (0.1–0.9) with ≥40% agreement threshold; iterative ToRA fallback for sub-step decomposition
- **Coding challenge subcategories**: 4 new subcategories (dp, ds, formal, sort_search) wired into max_tokens and stop sequences
- **README overhaul**: comprehensive architecture diagram, solver accuracy table, development history, project structure map, and Docker usage guide
- **GitHub push**: v12d branch with 8 new commits + uncommitted ToRA work pushed to `origin/v12d`

## Architecture Decisions

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| ADR-001 | Dispatcher profile as default for project work | Manager/dispatcher pattern was the single most productive workflow | 2026-07-13 |
| ADR-002 | Inform-and-act permission model | Permission-asking during hackathon crunch wasted momentum | 2026-07-13 |
| ADR-003 | Sub-agents pinned to deepseek-v4-flash | Consistent model behavior, fast inference | 2026-07-13 |
| ADR-004 | Worktree auto-commit on exit | Prevents lost work from accidental terminal close | 2026-07-13 |
| ADR-005 | CONTEXT.md → lean status + MEMORY.md/LOG.md for depth | CONTEXT.md grew to 239 lines of noise | 2026-07-13 |
