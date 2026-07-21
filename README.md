# Router to Vibehalla — AMD ACT II Hackathon (Track 1)

**Team:** Raiders of Vibehalla  
**Track:** Token-Efficient Routing Agent  
**Docker image:** `ghcr.io/artemkorolev1/amd-hackathon-submit`

A hybrid routing agent that classifies tasks across 8 categories and solves them using a pipeline of deterministic solvers and local LLM inference — all running on CPU within a 4 GB / 2 vCPU Docker container.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              /input/tasks.json                               │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           8-Way Classifier Cascade                           │
│        Primary: 8-way scorer (92.2%) + 4 secondary resolvers                 │
│    (code_debug↔code_gen, logic↔math, factual↔logic, summarization↔*)       │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│               Complexity Scoring (MiniLM + LogisticRegression)               │
│                        7-signal bitmorphic fallback                          │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Deterministic Solver Layer                           │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────────┐    │
│  │  NER v3   │ │   Logic   │ │   Math    │ │ Sentiment │ │   Factual    │    │
│  │ Regex+NLP │ │Proposition│ │ Template  │ │ Weighted  │ │    FactDB    │    │
│  │           │ │ Zebra, SAT│ │  Solvers  │ │ Keywords  │ │ (112K facts) │    │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └──────────────┘    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                             LLM Inference Layer                              │
│  ┌──────────────────────────────┐     ┌──────────────────────────────────┐   │
│  │    Qwen2.5-1.5B-Instruct     │     │   Qwen2.5-Coder-1.5B-Instruct    │   │
│  │     (general inference)      │     │   (code gen, math extraction)    │   │
│  └──────────────┬───────────────┘     └────────────────┬─────────────────┘   │
│                 │                                      │                     │
│                 └──────────────────┬───────────────────┘                     │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           ToRA Math Pipeline                           │  │
│  │    Variable extraction → Single-shot ToRA → Consensus voting           │  │
│  │    (4 temps, ≥40% agree) → Iterative fallback (sub-step decomp)        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            /output/results.json                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Solver Categories

| Category | Approach |
|----------|----------|
| Factual QA | Deterministic FTS5 knowledge base (112K facts) |
| Sentiment | Weighted keywords + VADER + cascade |
| NER | Regex + biomedical/general NER patterns |
| Code Generation | Template solvers + Qwen2.5-Coder-1.5B |
| Code Debugging | Pattern matching + LLM |
| Logical Reasoning | Propositional logic, zebra/BFS solvers |
| Math Reasoning | 11 template types + ToRA (planning + PythonExecutor) with consensus voting |
| Summarization | LLM with two-step entity extraction |

---

## Usage

```bash
# Pull CPU submission container
docker pull ghcr.io/artemkorolev1/amd-hackathon-submit:cpu

# Run with your tasks
docker run --rm \
  -v /path/to/tasks.json:/input/tasks.json:ro \
  -v /path/to/output:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:cpu
```

### Build Locally

```bash
make cpu-build    # docker buildx --platform linux/amd64 -f Dockerfile.cpu
make cpu-run      # run with /input /output mounts
```

### I/O Contract

- **Input:** `/input/tasks.json` — JSON array of task strings, or objects with a `"prompt"` key
- **Output:** `/output/results.json` — JSON array of answer strings
- **Stdout:** One answer per line (for pipe usage)
- **Stderr:** Log output (timestamps, categories, progress)

---

## Project Structure

```
├── agent/           # Core pipeline (classifier, cascade routers, solvers)
│   └── solvers/     # 40+ solver implementations (deterministic, ToRA, cascades)
├── staging/         # Parallel submission entrypoint (worker pool, judge)
├── data/facts/      # FTS5 knowledge base (112K facts)
├── Dockerfile.cpu   # CPU submission container (grader target)
├── Dockerfile       # Standard build
├── requirements.txt # Python dependencies
└── Makefile         # Build automation
```

---

## Constraints

| Constraint | Status |
|------------|--------|
| Max 10 GB compressed | ~4–5 GB expected |
| Startup < 60 s | ~15–20 s model load |
| Runtime < 10 min | ~2–6 min depending on tasks |
| CPU-only (2 vCPU, 4 GB RAM) | llama.cpp on CPU, 0 GPU layers |

---

## License

MIT
