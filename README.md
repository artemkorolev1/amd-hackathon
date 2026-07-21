# Router to Vibehalla — AMD ACT II Hackathon (Track 1)

**Team:** Raiders of Vibehalla  
**Track:** Token-Efficient Routing Agent  
**Deadline:** July 11, 2026

A hybrid routing agent that classifies tasks across 8 categories and solves them using a pipeline of deterministic solvers and local LLM inference — all running on CPU within a 4 GB / 2 vCPU Docker container.

> **Code repo (private):** [github.com/artemkorolev1/amd-hackathon-submit](https://github.com/artemkorolev1/amd-hackathon-submit)  
> **Docker images:** `ghcr.io/artemkorolev1/amd-hackathon-submit`

---

## What It Does

Reads a batch of tasks from `/input/tasks.json`, classifies each one into one of 8 categories, and solves it using the most efficient path available — deterministic pattern matching where possible, local LLM inference where needed.

### 8 Solver Categories

| Category | Approach |
|----------|----------|
| Factual QA | Deterministic FTS5 knowledge base (112K facts) |
| Sentiment | Weighted keywords + VADER + cascade |
| Named Entity Recognition | Regex + biomedical/general NER patterns |
| Code Generation | Template solvers + Qwen2.5-Coder-1.5B |
| Code Debugging | Pattern matching + LLM |
| Logical Reasoning | Propositional logic, zebra/BFS solvers |
| Math Reasoning | 11 template types + ToRA (planning + PythonExecutor) with consensus voting |
| Summarization | LLM-only with two-step entity extraction |

---

## Architecture (high level)

```
Input tasks → 8-way classifier cascade (92.2%) 
           → Complexity scoring (MiniLM + heuristics)
           → Deterministic solvers (5 categories)
           → LLM inference (Qwen2.5-1.5B + Qwen2.5-Coder-1.5B GGUF)
           → ToRA math pipeline (variable extraction → single-shot → consensus voting → iterative fallback)
           → /output/results.json
```

All inference runs locally via llama.cpp on CPU. No external API calls. Zero cloud dependencies.

---

## Docker

```bash
# Pull latest CPU submission container
docker pull ghcr.io/artemkorolev1/amd-hackathon-submit:cpu

# Run with your tasks
docker run --rm \
  -v /path/to/tasks.json:/input/tasks.json:ro \
  -v /path/to/output:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:cpu
```

### Image Tags

| Tag | Description |
|-----|-------------|
| `:cpu` | Latest CPU submission build (linux/amd64) |
| `:latest` | Latest full build (GPU-capable) |
| `:v12d` | v12d branch — GEPA-optimized prompts, cascade solvers |
| historical tags v0–v5 | Submission history during hackathon |

---

## Key Results

- **v12d Nemotron-3-Nano-4B:** **93.7%** on 300-question eval set (post-deadline)
- **Deterministic-only:** 36.0% on training (31.8% validation) — zero token cost
- **ToRA math pipeline:** 100% on training math, 84.2% on test set
- **FactDB:** 112K facts from integrated NQ-Open dataset
- **GEPA optimized prompts:** Genetic Pareto Algorithm tuned all 8 category prompts for min-token / max-accuracy

### Submission History

| Tag | Score | Notes |
|-----|-------|-------|
| v0 | 52.6% | Fireworks API + 2 solvers |
| v1 | Skipped | Fireworks removed |
| v2 | Runtime error | `parse_allowed_models()` crash |
| v3/v4 | 42.1% | Greedy deterministics stole tasks |
| v5 | Pending | Lowered Fireworks threshold to 0.10 |

---

## Constraints

| Constraint | Status |
|------------|--------|
| Max 10 GB compressed | ~4–5 GB expected |
| Startup < 60 s | ~15–20 s model load |
| Runtime < 10 min | ~2–6 min depending on tasks |
| CPU-only (2 vCPU, 4 GB RAM) | llama.cpp, 0 GPU layers |

---

## License

MIT
