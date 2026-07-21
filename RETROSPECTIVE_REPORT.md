# AMD ACT II Hackathon — Structured Retrospective Report

**Generated:** July 13, 2026  
**Scope:** Jul 6–13, 2026 (1 week sprint)  
**Project:** Router to Vibehalla / Track 1 Token Efficient Routing Agent  
**Team:** Raiders of Vibehalla  
**Hardware:** RTX A4000 8GB (GPU), 12c/30GB (dev), 2vCPU/4GB (Docker submission)  
**Deadline:** July 11, 2026 12:00 PM EDT — **MISSED**

---

## 1. Executive Summary

The AMD ACT II Hackathon was a highly ambitious 1-week sprint to build a token-efficient routing agent. The team built an extensive 5-stage pipeline (pre-filters → multi-axis features → category classification → complexity scoring → solver selection), integrated GPU-accelerated local inference with Fireworks API fallback, and iterated through 12+ container versions (v0–v12d). Despite achieving strong local eval results (93.7% on 300-set with v12d Nemotron-3-Nano-4B), the submission deadline was missed. The project suffered from worktree proliferation (6+ directories, 41GB), CPU overload from parallel agents crashing the dev machine, overly verbose context logging, and excessive permission-asking during crunch time. Major wins included the Hermes agent sub-agent dispatcher pattern, a robust eval pipeline with Pareto-optimized prompt evolution (GEPA), and successful model selection/tuning.

---

## 2. What Went Wrong (Issues)

### 2.1 Worktree Proliferation — "Million of different folders"

| Issue | Impact |
|-------|--------|
| 6+ hackathon directories: `amd-hackathon/`, `amd-hackathon-filtered-build/`, `amd-hackathon-v12h/`, `amd-hackathon-parallel/`, `amd-hackathon-shared/`, `amd-hackathon-submit/`, `amd-hackathon-k2p7/`, `amd-hackathon-v61/` | 41GB total disk usage, confusing git history |
| Multiple independent git repos with divergent histories (filtered-build and v12h were NOT worktrees — separate repos diverged from v9 onward) | Code changes scattered; couldn't find shared data across branches |
| Shared data files (eval_300.json, etc.) got overwritten when switching branches | Files "kept getting deleted" between worktrees |
| Symlink-based sharing between worktrees was fragile (broken symlinks when repos moved) | Intermittent data loss during cross-worktree operations |
| Git operations in one terminal affected all terminals in same directory | Confusion about "which branch am I on?" |

**Resolution:** Consolidated everything into main repo (~/dev/amd-hackathon/) on Jul 12. Created v12d branch. Deleted 4 redundant directories (freed 26GB). Created `hackathon-cleanup` skill to standardize future cleanups.

### 2.2 CPU Overload — System Freezes

| Symptom | Root Cause |
|---------|------------|
| System became unresponsive, mouse freezing | 3 parallel Hermes sub-agents + llama-cpp-python compilation (~20 cc1plus processes) + benchmark_ml_cascade.py ran simultaneously |
| Swap 8/8GB exhausted | Memory pressure caused kernel thrashing |
| Load average: 8.2 / 12.3 / 11.9 on 12 cores | Near 100% CPU saturation |
| 6 concurrent Hermes processes | No parallelism limits |

**Resolution:** CPU watchdog daemon built (cgroups v2 CPU quota), pre-flight resource gate added (load > 8 or swap > 90% blocks dispatch), max concurrent children reduced to 2, compilations run with `nice -n 19`.

### 2.3 CONTEXT.md Quality — "A lot of nonsense"

| Problem | Detail |
|---------|--------|
| File grew to 239 lines covering versions, submission log, decisions, blockers, ideas, future tasks, resource probe | Too much information, hard to find signal |
| Mixed active status with historical decision log | Every session appended without pruning |
| Protocol (START/DURING/END) encouraged append-only growth | No garbage collection |
| Multiple parallel sessions all writing to same file | Race conditions, duplicate entries |

**Insight:** The CONTEXT.md worked as a shared buffer but became unwieldy. A wiki or structured database would have been better.

### 2.4 Permission-Asking During Crunch

| Evidence | Context |
|----------|---------|
| CONTEXT.md line 153: "From now on: no pushes without explicit permission" | Jul 9 session — after Claude Code agents pushed without approval |
| User feedback: permission-asking was "too frequent" during the rush | During deadline crunch, every action requiring confirmation slowed momentum |
| The `context-md-logging` skill was designed to ask before writing | Added friction to cross-session logging |

**Resolution:** The user wanted fewer interruptions during crunch time, but the safety protocols (ask before push, ask before CONTEXT.md write) added overhead.

### 2.5 Missed Deadline — "Too ambitious"

| Factor | Detail |
|--------|--------|
| Scope creep | 5-stage pipeline, 6+ solvers, GEPA prompt evolution, GPU inference, pull architecture, staging system |
| Multiple architectural pivots | ML classifier → deterministic → hybrid → self-consistency voting → Nemotron-3-Nano-4B swap |
| Container iteration churn | 12+ versions (v0–v12d), multiple RUNTIME_ERROR submissions |
| Research over-build | 20+ competitor repos cloned and analyzed (27 git repos in hackathon_research/) |
| Complexities underestimated | Docker submission constraints (4GB RAM, 2vCPU, 10 min runtime) caused repeated redesigns |

**Timeline breakdown:**
- Jul 6-8: Initial setup, ML classifier, Fireworks integration
- Jul 9: v0-v5 submissions (52.6% → 42.1% → RUNTIME_ERROR)
- Jul 10: Architecture review, self-consistency voting plan, competitive analysis
- Jul 11: Data pipeline complete, Stage 0-1 built, GEPA architecture designed — **deadline missed**
- Jul 12: Folder cleanup, v12d consolidated
- Jul 13: GEPA prompt evolution, GPU evals, pipeline refinement (post-deadline)

---

## 3. What Worked (Successes)

### 3.1 Hermes Agent Performance — "Really Great"

| Strength | Evidence |
|----------|----------|
| Sub-agent delegation | Dispatched parallel sub-agents for research, code generation, eval runs, architecture review |
| Tool use | Git, Docker, file operations, web search, delegate_task all used extensively and correctly |
| Memory recall | Mnemosyne remembered worktree management rules, cleanup patterns, model swap decisions |
| Session continuity | CONTEXT.md protocol + cross-session awareness maintained coherence across 15+ sessions |

### 3.2 Dispatcher/Manager Pattern — Major Win

The user explicitly called out: **"Running not workers but managers and asking them to send teams for different tasks"** as the top success.

| Session | Pattern Used |
|---------|-------------|
| `Category Prompt Evolution Dispatcher` (Jul 13) | "This chat is a **dispatcher** — I design the prompt strategies, dispatch eval work to GPU sub-agents, get reports back, decide next moves, and repeat." |
| `Sub-Agent Pull System Build` (Jul 13) | "You are just a dispatcher run sub agents as much as possible give orders receive reports" |
| GEPA pipeline | Orchestrator → MutationAgent → EvaluationAgent → AnalysisAgent (3 agents, 3 libraries) |
| Architecture review | Claude Opus dispatched as senior architect to review implementation plan (found 5 critical issues) |

**Why it worked:** The dispatcher pattern prevented monolithic agent sessions from hitting context limits, allowed parallel GPU utilization, and kept the human-in-the-loop for decisions while offloading execution.

### 3.3 Eval Pipeline and Model Selection

| Achievement | Detail |
|------------|--------|
| 23-stage pipeline eval | Comprehensive evaluation pipeline tracking accuracy, latency, tokens |
| v12d: 93.7% on 300-set | Nemotron-3-Nano-4B achieved best local accuracy (281/300) |
| Math accuracy: 0.021 → 0.564 | 13× improvement from max_tokens fix + 3-step workflow (Llama-3.2) |
| GEPA prompt evolution | Genetic Pareto Algorithm evolved prompts across 5 core tasks (factual, math, sentiment, summarization, NER) |
| Competitive analysis | Analyzed top 10 leaderboard submissions, adopted Frugal Router's self-consistency voting and circuit breaker patterns |
| Dataset pipeline | 79K items labeled, 4 stages of train/val/test splits, 3 pipeline bugs fixed recovering 58K items |

### 3.4 Shared Data Consolidation

| Win | Detail |
|-----|--------|
| Shared data directory created | `/home/artem/dev/amd-hackathon-shared/` as symlink target |
| Symlink + skip-worktree pattern | Data files accessible across all worktrees without git conflicts |
| 3 worktrees linked | classifier-experiment-4way, v6.1-k2p7, v6.1-baseline all saw same data |
| Consolidation script | `consolidate-all-worktrees.sh` automated the linking |

### 3.5 Docker Infrastructure

| Win | Detail |
|-----|--------|
| Production Dockerfile | python:3.12-slim, 472MB, GHCR public package |
| GPU override support | Local testing with `--gpus all`, grader CPU-only auto-fallback |
| Circuit breaker pattern | 5 consecutive failure limit, 30s retry, prevents Fireworks API cascade failures |
| Time budget degradation | 50/70/85% thresholds gracefully degrade sampling as deadline approaches |

---

## 4. Key Metrics

### 4.1 Container Submission History

| Version | Score | Cause |
|---------|-------|-------|
| v0-submitted | 52.6% | Original Fireworks + 2 solvers |
| v1-no-fireworks | Skipped | `:latest` overwritten before grader |
| v2-fireworks-030 | RUNTIME_ERROR | `parse_allowed_models()` TypeError |
| v3-parse-fix | 42.1% | Greedy deterministics stole Fireworks tasks |
| v4 (same as v3) | 42.1% | Duplicate tag |
| v5-threshold-010 | PENDING | Lowered Fireworks threshold 0.30→0.10 |
| v12d (Nemotron) | 93.7% (local) | Never submitted — deadline missed |

### 4.2 Sessions Count

- ~15 tracked Hermes sessions over 8 days
- Key sessions: Jul 6 (prep), Jul 9 (v0-v5 submissions), Jul 10 (architecture review), Jul 11 (data pipeline + GEPA), Jul 12 (cleanup), Jul 13 (GPU evals)

### 4.3 Repositories Used

- 27 research repos cloned (competitor analysis)
- 1 main submission repo (amd-hackathon-submit)
- 1 public placeholder repo (amd-hackathon)
- 6 local worktree directories (peak)

---

## 5. Recommendations for Next Hackathon

### 5.1 Do This Again
- **Dispatcher/manager pattern** — Architect the AI agent as a dispatcher from day 1. The manager sends sub-agents for specific tasks and synthesizes results.
- **Eval-first methodology** — Build the eval pipeline before the solver. v12d's 93.7% accuracy came from systematic evaluation.
- **Shared data consolidation** — Single source of truth outside git (symlinked into worktrees).
- **Competitive analysis early** — Researching winning repos (Frugal Router, etc.) provided proven patterns.

### 5.2 Change This
- **One repo, not six** — Use git worktrees (Hermes built-in `-w` flag) for parallel branches, not separate repos with copies of data.
- **Stop building before submitting** — Ship a minimal viable submission on day 2, iterate from there. The team spent 4+ days building infrastructure and never shipped v12d.
- **CPU quotas from day 1** — cgroups v2 CPU limits should be set before dispatching any sub-agents.
- **Prune CONTEXT.md weekly** — Archive old sections, keep only last 48h of active state.
- **Reduce permission gates during crunch** — Have a "crunch mode" flag that relaxes confirmation requirements.

### 5.3 Start Doing
- **Time-boxing for each version** — 2 hours max per container iteration. If it's not shipping, cut scope.
- **Pre-flight resource checks** — Check load/swap/RAM before any expensive operation.
- **Rolling eval baselines** — Track accuracy vs tokens vs latency in a spreadsheet across versions.
- **Hard deadline for architecture decisions** — Freeze architecture by day 3, only bugfix after.

---

## 6. Appendix: Session Inventory

| Session ID | Title | Key Topic |
|-----------|-------|-----------|
| 20260713_080558_68c97f | Dispatcher manager continuation | Final push, GPU evals, sub-agent orchestration |
| 20260713_080235_ea091f | Commit and Push Summary | Git cleanup, 104 files committed |
| 20260713_071855_b31ef2 | CPU Container Preparation | Docker CPU/GPU dual-container setup |
| 20260713_063513_eb7c34 | Clarifying Evaluation Objectives | GEPA multi-category eval workflows |
| 20260713_060410_4e58c7 | Sub-Agent Pull System Build | Dispatcher pattern, pull architecture |
| 20260713_052251_ad5491 | Controlling CPU Consumption | CPU overload analysis, watchdog daemon |
| 20260713_032156_0a9738 | Category Prompt Evolution Dispatcher | **Key dispatcher session** — GPU sub-agent eval orchestration |
| 20260712_174852_1f591b | Hackathon Folder Cleanup Plan | Worktree consolidation, 4 dirs deleted |
| 20260711_083833_4ad4b2 | Aligned for Stage One | Stage 1 multi-axis features built |
| 20260711_083619_aff1a9 | Hackathon Folder Explanation | Repo architecture explained, deterministic.py split |
| 20260711_052215_92e3ec | Super Eval Framework Rework | Data consolidation, pipeline audit, 3 bugs fixed |
| 20260710_181857_9b548e | Automated Context Logging Across Chats | CONTEXT.md protocol, worktree management skill |
| 20260710_181025_ac936d | Sub Agent Project Pattern Discovery | Competitive analysis, Frugal Router patterns |
| 20260709_* | v0-v5 submissions | Container submissions, 52.6% → RUNTIME_ERROR |
| 20260706_204835_58b735 | Initial Setup | Project kickoff, pre-hackathon prep |
