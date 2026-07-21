# Langfuse Integration Plan — AMD ACT II Pipeline

> **Target:** Docker-based Python pipeline (`python:3.12-slim`, 4 GB RAM, 2 vCPU)
> **Pipeline:** Input → pre_filter → classify (8-way) → complexity_filter → route → {deterministic | local GGUF | Fireworks API} → post_processor → grader
> **Goal:** Full observability, per-node evaluation, dataset management, and comparison dashboards

---

## Table of Contents

1. [Self-Hosted Deployment](#1-self-hosted-deployment)
2. [Per-Node Instrumentation](#2-per-node-instrumentation)
3. [Node Versioning](#3-node-versioning)
4. [Section Versioning](#4-section-versioning)
5. [Dataset Management](#5-dataset-management)
6. [Per-Node Evaluation](#6-per-node-evaluation)
7. [Eval Run Comparison](#7-eval-run-comparison)
8. [Custom Dashboards](#8-custom-dashboards)
9. [Implementation Checklist](#9-implementation-checklist)

---

## 1. Self-Hosted Deployment

### Minimal Docker Compose (add to your existing compose file)

```yaml
# docker-compose.yml — add alongside your pipeline service
version: "3.9"

services:
  # ── Your existing pipeline ──────────────────────────────────────
  pipeline:
    build: .
    image: ghcr.io/artemkorolev1/amd-hackathon-submit
    container_name: amd-pipeline
    environment:
      - LANGFUSE_HOST=http://langfuse:3000
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-pk-local}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-sk-local}
      - LANGFUSE_ENABLE=1
    depends_on:
      langfuse:
        condition: service_healthy
    volumes:
      - /input:/input:ro
      - /output:/output
      - ./models:/models:ro
    # ... rest of your pipeline env vars

  # ── Langfuse (self-hosted, no MinIO/Redis/Prometheus) ──────────
  langfuse:
    image: ghcr.io/langfuse/langfuse:latest
    container_name: langfuse
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_LANGFUSE_HOST=http://localhost:3000
      - DATABASE_URL=postgresql://langfuse:langfuse@postgres:5432/langfuse
      - LANGFUSE_ENABLE_EXPORT=1
      # Minimal config — no S3, no Redis, no Prometheus
      - S3_ENABLED=false
      - REDIS_ENABLED=false
      - METRICS_DISABLED=true
      # Auto-migrate DB on startup
      - DATABASE_MIGRATION_ENABLED=true
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "http://localhost:3000/api/public/health"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 30s
    depends_on:
      postgres:
        condition: service_healthy

  # ── Postgres (Langfuse's only dependency) ───────────────────────
  postgres:
    image: postgres:16-alpine
    container_name: langfuse-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=langfuse
      - POSTGRES_PASSWORD=langfuse
      - POSTGRES_DB=langfuse
    volumes:
      - langfuse-pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  langfuse-pgdata:
```

### Resource Budget

| Component | Image | RAM | CPU | Disk |
|-----------|-------|-----|-----|------|
| Langfuse | ghcr.io/langfuse/langfuse | ~256 MB | 0.2 vCPU | — |
| PostgreSQL | postgres:16-alpine | ~128 MB | 0.1 vCPU | ~1 GB eval data |
| **Total overhead** | | **~384 MB** | **~0.3 vCPU** | |

Well within the 4 GB / 2 vCPU budget.

### Critical Setup Steps

1. **Initial access:** After `docker compose up`, visit `http://localhost:3000`
2. **Create initial user** (first-registration flow in Langfuse UI)
3. **Copy API keys:** Settings → API Keys → Create `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
4. **Set env vars** in your `docker-compose.yml` or `.env` file

---

## 2. Per-Node Instrumentation

### Python Dependencies

```bash
# Add to requirements.txt
langfuse>=2.55.0
openai>=1.0.0         # already present via fireworks
```

### Langfuse Client Initialization

Create a shared module `agent/langfuse_client.py`:

```python
"""agent/langfuse_client.py — Shared Langfuse client singleton."""

import os
from langfuse import Langfuse

_langfuse: Langfuse | None = None

def get_langfuse() -> Langfuse | None:
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    if not os.environ.get("LANGFUSE_ENABLE", "0") == "1":
        return None
    _langfuse = Langfuse(
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-local"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", "sk-local"),
        sdk_integration="amd-hackathon-pipeline",
    )
    return _langfuse
```

### @observe Decorator Placements

Instrument **each** pipeline node individually. Below are the exact functions to wrap and the metadata to report.

#### Node 1: pre_filter

**File:** `agent/pre_filter.py`
**Function:** `stage0(prompt: str) -> Stage0Result`

```python
from agent.langfuse_client import get_langfuse

@observe(name="pre_filter_v1", capture_input=True, capture_output=True)
def stage0(prompt: str) -> Stage0Result:
    # existing implementation ...
```

**Metadata to report:**
```python
langfuse_context.update_current_trace(
    input=prompt,
    metadata={
        "node": "pre_filter",
        "node_version": "pre_filter-v1",
        "prompt_length": len(prompt),
        "has_code_fence": bool(re.search(r"```", prompt)),
        "has_greeting": bool(RE_GREETING.match(prompt)),
    }
)
```
After computing result, add output metadata:
```python
langfuse_context.update_current_observation(
    output=result,
    metadata={
        "node": "pre_filter",
        "action": result.action,
        "direct_category": result.category,
        "flags": result.flags,
        "bypassed": result.action in ("bypass", "route_to_stage3"),
    }
)
```

---

#### Node 2: classify (8-way classifier)

**File:** `agent/category_filter.py`
**Function:** `classify_with_detail(prompt: str) -> dict`

```python
@observe(name="classifier_v1", capture_input=True, capture_output=True)
def classify_with_detail(prompt: str) -> dict:
    # existing implementation ...
```

**Metadata:**
```python
langfuse_context.update_current_observation(
    input=prompt,
    metadata={
        "node": "classifier",
        "node_version": "classifier-v1",
        "categories": result.get("scores", {}),
        "winner": result.get("category"),
        "confidence": result.get("confidence"),
        "runner_up": sorted(result.get("scores",{}).items(), key=lambda x:-x[1])[1][0] if len(result.get("scores",{})) > 1 else None,
        "margin": sorted(result.get("scores",{}).values(), reverse=True)[0] - sorted(result.get("scores",{}).values(), reverse=True)[1] if len(result.get("scores",{})) > 1 else 0,
    }
)
```

---

#### Node 3: complexity_filter

**File:** `agent/complexity_filter.py`
**Function:** `score(prompt: str, category: str) -> float`

```python
@observe(name="complexity_filter_v1", capture_input=True, capture_output=True)
def score(prompt: str, category: str) -> float:
    # existing implementation ...
```

**Metadata:**
```python
langfuse_context.update_current_observation(
    input={"prompt": prompt, "category": category},
    output=complexity_score,
    metadata={
        "node": "complexity_filter",
        "node_version": "complexity_filter-v1",
        "category": category,
        "complexity_score": complexity_score,
        "simple_max_threshold": COMPLEXITY_THRESHOLDS.get("simple_max", 0.3),
    }
)
```

---

#### Node 4: route (decision table)

**File:** `agent/main.py`
**Function:** No standalone function — inline in `_run_pipeline()`. Create a wrapper.

**New wrapper in `agent/routing.py`:**

```python
# agent/routing.py — extract the decision logic

@observe(name="router_v1", capture_input=True, capture_output=True)
def route_decision(
    category: str,
    complexity: float,
    scores: dict,
    deterministic_categories: set,
    naked_categories: set,
    complexity_thresholds: dict,
) -> dict:
    """Return {'action': 'deterministic'|'api', 'model_type': ...}"""
    # existing decision logic from main.py
    ...
```

**Metadata:**
```python
langfuse_context.update_current_observation(
    input={
        "category": category,
        "complexity": complexity,
        "top_category": top_cat,
        "top_score": top_score,
        "second_score": second_score,
    },
    output=decision,
    metadata={
        "node": "router",
        "node_version": "router-v1",
        "decision": decision["action"],
        "model_type": decision.get("model_type"),
        "is_merged_prompt": use_merged,
        "reason": (
            f"simple+deterministic" if decision["action"] == "deterministic"
            else f"api_needed"
        ),
    }
)
```

---

#### Node 5a: deterministic solver

**File:** `agent/solvers/deterministic.py`
**Functions:** `solve_arithmetic`, `solve_logic`, `solve_sentiment`, `solve_ner`, `solve_factual_qa`, `solve_code_debugging`, `solve_summarization`

Create a **single** wrapper that routes to the appropriate solver:

```python
# agent/solvers/deterministic.py — add wrapper

@observe(name="deterministic_solver_v1", capture_input=True, capture_output=True)
def solve_deterministic(prompt: str, category: str) -> str | None:
    solver_map = {
        "math_arithmetic": solve_arithmetic,
        "logical_reasoning": solve_logic,
        "sentiment": solve_sentiment,
        "named_entity_recognition": solve_ner,
        "other_complex": solve_factual_qa,
        "code_debugging": solve_code_debugging,
        "summarization": solve_summarization,
    }
    solver_fn = solver_map.get(category)
    if not solver_fn:
        return None
    return solver_fn(prompt, category)
```

**Metadata:**
```python
langfuse_context.update_current_observation(
    input={"prompt": prompt, "category": category},
    metadata={
        "node": "deterministic_solver",
        "node_version": "deterministic_solver-v1",
        "solver_type": category,
        "solver_fn": solver_fn.__name__,
    }
)
```

---

#### Node 5b: local GGUF solver

**File:** `agent/solvers/local_vote.py`
**Function:** `solve_with_consensus(llm, prompt, category, ...) -> dict`

```python
@observe(name="local_gguf_solver_v1", capture_input=True, capture_output=True)
def solve_with_consensus(llm, prompt, category, system_prompt, k, max_tokens):
    # existing implementation ...
```

**Metadata:**
```python
langfuse_context.update_current_observation(
    input={"prompt": prompt, "category": category},
    metadata={
        "node": "local_gguf_solver",
        "node_version": "local_gguf_solver-v1",
        "model": os.environ.get("MODEL_PATH", "unknown"),
        "consensus_samples": k,
        "category": category,
    },
    output={
        "answer": result.get("majority_answer"),
        "agreement_score": result.get("agreement_score"),
        "sample_count": k,
    }
)
```

---

#### Node 5c: Fireworks API solver

**File:** `agent/solvers/fireworks.py`
**Function:** `solve(self, model, user_prompt, system_prompt, max_tokens, task_type, det_hint)`

```python
# Inside FireworksSolver class

@observe(name="fireworks_solver_v1", capture_input=True, capture_output=True)
def solve(self, model, user_prompt, system_prompt, max_tokens, task_type, det_hint):
    # existing implementation ...
```

**Metadata:**
```python
langfuse_context.update_current_observation(
    input=user_prompt,
    metadata={
        "node": "fireworks_solver",
        "node_version": "fireworks_solver-v1",
        "model": model,
        "task_type": task_type,
        "max_tokens": max_tokens,
        "has_det_hint": bool(det_hint),
        "system_prompt_length": len(system_prompt or ""),
    },
    output=answer,
)
```

---

#### Node 6: post_processor (QC gate)

**File:** `agent/solvers/verify.py`
**Function:** `verify(answer, category, task) -> VerifyResult`

```python
@observe(name="post_processor_v1", capture_input=True, capture_output=True)
def verify(answer, category="", task=""):
    # existing implementation ...
```

**Metadata:**
```python
langfuse_context.update_current_observation(
    input={"answer": answer, "category": category},
    metadata={
        "node": "post_processor",
        "node_version": "post_processor-v1",
        "category": category,
        "answer_length": len(answer or ""),
        "passed": result.passed,
        "failure_reason": result.reason if not result.passed else None,
    }
)
```

---

### Pipeline-Level Trace

Wrap the entire `process` method in `agent/pipeline.py` or `agent/main.py` as a **trace** that contains all the above observation spans:

```python
from langfuse.decorators import observe, langfuse_context

def process_task(self, task_id: str, prompt: str) -> str:
    with get_langfuse().trace(
        name="pipeline_run",
        input=prompt,
        metadata={
            "task_id": task_id,
            "pipeline_version": "pipeline-v7-complexity",
            "section_versions": {
                "solver_group": "solver_group-v2",
                "classifier_group": "classifier_group-v1",
            },
        },
    ) as trace:
        # Each node call will auto-link to this trace via @observe
        # or you can pass trace_id explicitly
        ...
```

The `@observe` decorator on each function auto-creates child spans under the current trace context.

### Latency & Token Tracking

Langfuse automatically records:
- **Duration** of each `@observe`-wrapped function (millisecond precision)
- **Token usage** for LLM calls (if you pass `usage` dict to the observation)
- **Error count** (exceptions raised in observed functions)

Add explicit token tracking to the Fireworks and local GGUF nodes:

```python
# After fireworks.solve() call:
langfuse_context.update_current_observation(
    usage={
        "input": len(user_prompt.split()) * 1.3,   # approximate
        "output": len(answer.split()),
        "unit": "TOKENS",
    }
)
```

For accuracy, parse the actual token counts from the API response.

---

## 3. Node Versioning

### Strategy

Each node gets a **semantic version label** embedded in the `@observe(name=...)` decorator. Changing the version creates a new "generation" in Langfuse that appears as a distinct deployment in the UI.

| Node | Initial Version | File | `@observe` Name |
|------|----------------|------|-----------------|
| pre_filter | `pre_filter-v1` | `agent/pre_filter.py` | `@observe(name="pre_filter_v1")` |
| classifier | `classifier-v1` | `agent/category_filter.py` | `@observe(name="classifier_v1")` |
| complexity_filter | `complexity_filter-v1` | `agent/complexity_filter.py` | `@observe(name="complexity_filter_v1")` |
| router | `router-v1` | `agent/routing.py` | `@observe(name="router_v1")` |
| deterministic_solver | `deterministic_solver-v1` | `agent/solvers/deterministic.py` | `@observe(name="deterministic_solver_v1")` |
| local_gguf_solver | `local_gguf_solver-v1` | `agent/solvers/local_vote.py` | `@observe(name="local_gguf_solver_v1")` |
| fireworks_solver | `fireworks_solver-v1` | `agent/solvers/fireworks.py` | `@observe(name="fireworks_solver_v1")` |
| post_processor | `post_processor-v1` | `agent/solvers/verify.py` | `@observe(name="post_processor_v1")` |

### How to Change a Version

When you modify a node, update the version string in the decorator:

```python
# Before
@observe(name="classifier_v1", ...)

# After (tweaked scoring, improved regexes)
@observe(name="classifier_v2", ...)
```

Or parameterize via environment variable:

```python
# More flexible approach — reference env var in version string
NODE_VERSION = os.environ.get("CLASSIFIER_VERSION", "classifier-v1")

@observe(name=NODE_VERSION, capture_input=True, capture_output=True)
def classify_with_detail(prompt: str) -> dict:
    ...
```

Env-based approach allows A/B testing without code changes:
```bash
# Run A
CLASSIFIER_VERSION=classifier-v1 docker compose up

# Run B
CLASSIFIER_VERSION=classifier-v2 docker compose up
```

### Version Tracking in Metadata

Add version info to every observation's metadata for downstream querying:

```python
langfuse_context.update_current_observation(
    metadata={
        "node": "classifier",
        "node_version": "classifier-v1",  # explicit in metadata
        "deploy_time": "2026-07-13T12:00:00Z",
        "git_sha": "e2932cd",
    }
)
```

---

## 4. Section Versioning

### What is a Section?

A **section** is a group of nodes that are versioned atomically. Changing any node in a section bumps the section version.

### Defined Sections for This Pipeline

| Section Name | Member Nodes | Rationale |
|-------------|--------------|-----------|
| `classifier_group` | pre_filter, classifier | Pre-filter and classifier work together — changing pre-filter rules affects what reaches the classifier |
| `solver_group` | deterministic_solver, local_gguf_solver, fireworks_solver | Solvers are interchangeable — you often tweak all three together |
| `quality_group` | post_processor | Standalone — QC thresholds don't affect other nodes |
| `routing_group` | complexity_filter, router | Complexity scoring and routing decisions are tightly coupled |

### Section Version Manifest

Create `agent/section_versions.py`:

```python
"""agent/section_versions.py — Atomic section versioning."""

import os

# Read section versions from env (with code defaults)
# When you bump any node in a section, bump the section version too.

SECTION_VERSIONS = {
    "classifier_group": os.environ.get(
        "SECTION_VERSION_CLASSIFIER", "classifier_group-v1"
    ),
    "solver_group": os.environ.get(
        "SECTION_VERSION_SOLVER", "solver_group-v1"
    ),
    "quality_group": os.environ.get(
        "SECTION_VERSION_QUALITY", "quality_group-v1"
    ),
    "routing_group": os.environ.get(
        "SECTION_VERSION_ROUTING", "routing_group-v1"
    ),
}


def get_section_meta() -> dict:
    return {
        "section_versions": dict(SECTION_VERSIONS),
    }
```

### How to Bump a Section

When you modify `category_filter.py` (classifier node):

```bash
# 1. Bump the node version
export CLASSIFIER_VERSION=classifier-v3

# 2. Bump the section version (because classifier_group changed)
export SECTION_VERSION_CLASSIFIER=classifier_group-v3

# 3. Run
docker compose up
```

### Section Metadata in Traces

Include section versions in the top-level trace metadata:

```python
trace = langfuse.trace(
    name="pipeline_run",
    metadata={
        **get_section_meta(),
        "task_id": task_id,
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
    }
)
```

### Comparing Section Versions in Queries

Langfuse allows filtering by trace metadata:

```
# Langfuse query API
GET /api/public/traces?metadata.section_versions.solver_group=solver_group-v2
```

Use this to compare runs across section versions on dashboards.

---

## 5. Dataset Management

### Dataset Structure

Create **three** datasets in the Langfuse UI (or via the SDK at startup):

#### Dataset 1: `pipeline_runs` — Every prompt + routing decision + outcome

| Field | Source | Type | Example |
|-------|--------|------|---------|
| `prompt` | Input | string | "Fix bug: ..." |
| `task_id` | Input | string | "idx_42" |
| `category` | classifier output | string | "code_debug" |
| `classifier_confidence` | classifier output | float | 0.85 |
| `complexity_score` | complexity_filter output | float | 0.42 |
| `route_decision` | router output | string | "deterministic" / "api" |
| `solver_used` | actual solver used | string | "arithmetic" / "qwen2.5-1.5b" / "fireworks-kimi-k2p6" |
| `answer` | final output | string | "The bug is..." |
| `answer_length` | post_processor | int | 142 |
| `qc_passed` | post_processor | bool | true |
| `qc_reason` | post_processor | string | null |
| `pipeline_version` | trace metadata | string | "pipeline-v7-complexity" |
| `section_versions` | trace metadata | json | {...} |
| `latency_ms` | calculated | float | 1234.5 |
| `token_count` | solver | int | 456 |
| `timestamp` | auto | datetime | ... |

#### Dataset 2: `misclassifications` — Labeled classification errors

Same schema as `pipeline_runs` plus:

| Field | Source | Type | Example |
|-------|--------|------|---------|
| `corrected_category` | Manual review | string | "math" |
| `original_category` | classifier output | string | "logic" |
| `tag` | Manual review | string | "misclassification" |
| `notes` | Manual review | string | "Confused ratio problem for logic" |

#### Dataset 3: `eval_results` — Ground-truth annotations

| Field | Source | Type | Example |
|-------|--------|------|---------|
| `task_id` | Input | string | "idx_42" |
| `prompt` | Input | string | "What is 2+2?" |
| `category` | classifier | string | "math" |
| `ground_truth_category` | Human annotation | string | "math" |
| `ground_truth_answer` | Human annotation | string | "4" |
| `pipeline_answer` | Pipeline output | string | "4" |
| `category_correct` | Calculated | bool | true |
| `answer_correct` | Calculated | bool | true |
| `eval_run_id` | Eval run | string | "eval-20260713" |

### Creating Datasets via SDK

```python
# agent/dataset_manager.py

from agent.langfuse_client import get_langfuse

def ensure_datasets():
    lf = get_langfuse()
    if lf is None:
        return
    for name in ["pipeline_runs", "misclassifications", "eval_results"]:
        try:
            lf.create_dataset(name=name, description=...)
        except Exception:
            pass  # already exists
```

### Inserting Items During Pipeline Run

Add to `harness.py` or `agent/main.py` after each task completes:

```python
from agent.langfuse_client import get_langfuse

def _log_to_dataset(task_id, prompt, result, metadata):
    lf = get_langfuse()
    if lf is None:
        return
    lf.dataset_item.create(
        dataset_name="pipeline_runs",
        input={"task_id": task_id, "prompt": prompt},
        expected_output=result.get("answer"),
        metadata=metadata,
    )
```

### Weekly Misclassification Export

Create a script `scripts/export_misclassifications.py`:

```python
#!/usr/bin/env python3
"""Export misclassifications from Langfuse to CSV for review."""

import csv
import os
from datetime import datetime, timedelta
from agent.langfuse_client import get_langfuse

def main():
    lf = get_langfuse()
    if lf is None:
        return

    week_ago = datetime.utcnow() - timedelta(days=7)

    # Fetch misclassification dataset items
    items = lf.get_dataset("misclassifications").items

    with open("/output/misclassifications_weekly.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "prompt", "original_category", "corrected_category",
            "confidence", "tag", "notes", "latency_ms", "timestamp"
        ])
        writer.writeheader()
        for item in items:
            if item.created_at < week_ago:
                continue
            writer.writerow({
                "task_id": item.input.get("task_id"),
                "prompt": item.input.get("prompt"),
                "original_category": item.metadata.get("original_category"),
                "corrected_category": item.expected_output.get("corrected_category"),
                "confidence": item.metadata.get("classifier_confidence"),
                "tag": item.metadata.get("tag", "misclassification"),
                "notes": item.metadata.get("notes", ""),
                "latency_ms": item.metadata.get("latency_ms"),
                "timestamp": item.created_at.isoformat(),
            })

    print(f"Exported {len(items)} misclassifications to /output/misclassifications_weekly.csv")

if __name__ == "__main__":
    main()
```

Add a cron-like trigger in your Docker Compose (or run manually weekly):

```yaml
# Add as a periodic container
misclassification-export:
  build: .
  entrypoint: ["python", "-u", "/scripts/export_misclassifications.py"]
  profiles: ["cron"]
  depends_on:
    - langfuse
```

### Tagging Misclassifications in the UI

1. Open Langfuse UI → Datasets → `misclassifications`
2. For each item, click "Edit" → set `expected_output.corrected_category` to the correct category
3. Add `tag: "misclassification"` in metadata
4. The weekly export script picks up all tagged items

---

## 6. Per-Node Evaluation

### Evaluation Model

Langfuse evaluates by comparing **actual output** to **expected output** (ground truth). For per-node eval, we need ground-truth datasets per node.

### Eval Configurations

Create evaluations for each node:

#### 6.1 Classifier Accuracy Eval

```python
# evals/classifier_accuracy.py

from langfuse import Langfuse
import json

def eval_classifier():
    lf = Langfuse(...)

    # Define the eval
    classifier_eval = lf.create_eval_config(
        name="classifier_accuracy",
        description="8-way category classifier accuracy",
        scoring_type="BOOLEAN",  # correct/incorrect
        metadata={
            "node": "classifier",
            "measure": "accuracy",
        }
    )

    # Fetch dataset items with ground-truth categories
    dataset = lf.get_dataset("eval_results")

    for item in dataset.items:
        if not item.expected_output.get("ground_truth_category"):
            continue

        actual_category = item.metadata.get("category", "")
        expected_category = item.expected_output["ground_truth_category"]
        is_correct = actual_category == expected_category

        # Create a score run
        lf.score(
            trace_id=item.metadata.get("trace_id"),
            name="classifier_accuracy",
            value=1.0 if is_correct else 0.0,
            comment=(
                f"Expected {expected_category}, got {actual_category}"
                if not is_correct else "Correct"
            ),
            data_type="BOOLEAN",
            config_id=classifier_eval.id,
        )

    # Aggregated result
    print(f"Classifier accuracy: {aggregate_score}")
```

#### 6.2 Complexity Filter Accuracy Eval

```python
def eval_complexity():
    lf = Langfuse(...)
    complexity_eval = lf.create_eval_config(
        name="complexity_accuracy",
        description="Complexity score correctness (threshold-based)",
        scoring_type="NUMERIC",
        metadata={"node": "complexity_filter"},
    )
    # Compare complexity score against human-annotated complexity (0 or 1)
    ...
```

#### 6.3 Solver Accuracy Eval

```python
def eval_solver():
    lf = Langfuse(...)
    solver_eval = lf.create_eval_config(
        name="solver_accuracy",
        description="Answer correctness per solver type",
        scoring_type="BOOLEAN",
        metadata={"node": "solver"},
    )
    # Compare pipeline_answer vs ground_truth_answer
    # Group by solver_used (deterministic / local_gguf / fireworks)
    ...
```

#### 6.4 Router Decision Accuracy Eval

```python
def eval_router():
    lf = Langfuse(...)
    router_eval = lf.create_eval_config(
        name="router_decision_accuracy",
        description="Was the correct solver selected?",
        scoring_type="BOOLEAN",
        metadata={"node": "router"},
    )
    # Compare route_decision against optimal decision for given category/complexity
    ...
```

#### 6.5 Post-Processor Gate Accuracy Eval

```python
def eval_post_processor():
    lf = Langfuse(...)
    postproc_eval = lf.create_eval_config(
        name="qc_gate_accuracy",
        description="Did QC correctly accept/reject?",
        scoring_type="BOOLEAN",
        metadata={"node": "post_processor"},
    )
    # Validate: for known-good answers, did QC pass?
    # For known-bad answers, did QC reject?
    ...
```

### Running Evaluations

Schedule these as one-off scripts or periodic jobs:

```bash
# Run all evals
python evals/classifier_accuracy.py
python evals/solver_accuracy.py
python evals/router_decision_accuracy.py

# View results in Langfuse UI: Evaluations tab
```

---

## 7. Eval Run Comparison

### Creating Eval Runs

Use Langfuse's **experiment tracking** to group related runs:

```python
def create_eval_run(lf, run_name: str, params: dict):
    """Create a new eval run with parameter snapshot."""
    # Fetch existing params
    base_params = {
        "classifier_version": "classifier-v1",
        "solver_version": "solver_group-v1",
        "fireworks_threshold": 0.30,
        "complexity_threshold_simple": 0.3,
    }

    return lf.create_dataset_run(
        name=run_name,
        params={**base_params, **params},
        description=f"Comparison run: {run_name}",
    )
```

### Naming Convention for Eval Runs

```
eval-YYYYMMDD-{variant}-{version}
```

Examples:
- `eval-20260713-classifier-v1-baseline`
- `eval-20260713-classifier-v2`
- `eval-20260713-solver-group-v2`
- `eval-20260714-fireworks-threshold-010`

### Side-by-Side Comparison

In the Langfuse UI:

1. **Models tab** → Compare → Select two eval runs
2. View:
   - Latency per node (side-by-side box plots)
   - Accuracy per category (before/after)
   - Routing distribution (% deterministic vs % API)

### Programmatic Comparison via SDK

```python
# compare_runs.py

from agent.langfuse_client import get_langfuse

def compare_runs(run_a: str, run_b: str):
    lf = get_langfuse()

    scores_a = lf.fetch_scores(run_name=run_a)
    scores_b = lf.fetch_scores(run_name=run_b)

    print(f"{'Metric':<30} {'Run A':>10} {'Run B':>10} {'Delta':>10}")
    print("-" * 60)

    metrics = ["classifier_accuracy", "solver_accuracy", "avg_latency_ms"]
    for metric in metrics:
        va = scores_a.get(metric, 0)
        vb = scores_b.get(metric, 0)
        delta = vb - va
        print(f"{metric:<30} {va:>10.3f} {vb:>10.3f} {delta:>+10.3f}")

    print(f"\n--- Category-Level Comparison ---")
    categories = ["math", "code_debug", "sentiment", "ner", "logic", "factual", "summarization", "code_gen"]
    for cat in categories:
        ca = scores_a.get(f"accuracy_{cat}", 0)
        cb = scores_b.get(f"accuracy_{cat}", 0)
        delta = cb - ca
        print(f"{cat:<30} {ca:>10.3f} {cb:>10.3f} {delta:>+10.3f}")
```

### Tracking Parameter Changes Over Time

Create a **drift-tracking dashboard** that plots:

- **Classifier accuracy** per week (line chart with per-category series)
- **Average latency per node** per week
- **Routing distribution**: % deterministic vs % local GGUF vs % Fireworks
- **Token usage** per batch (total and per-category)
- **QC pass rate** per solver type

---

## 8. Custom Dashboards

### Dashboard 1: Per-Category Accuracy Trends

**Type:** Time-series line chart
**X-axis:** Date (daily/weekly buckets)
**Y-axis:** Accuracy (0.0–1.0)
**Series:** One line per category (8 categories)
**Source:** Scores from `classifier_accuracy` eval
**Query:**
```
traces | where metadata.section_versions.classifier_group contains "classifier_group"
| eval accuracy = if(scores.classifier_accuracy == 1, 1, 0)
| stats avg(accuracy) by category=metadata.category, _time
```

### Dashboard 2: Node-Level Latency

**Type:** Box plot
**X-axis:** Node name (pre_filter, classifier, complexity_filter, router, deterministic_solver, local_gguf_solver, fireworks_solver, post_processor)
**Y-axis:** Latency in milliseconds (log scale)
**Color:** By node version
**Source:** Observation durations from traces
**Query:**
```
observations | where name in ("pre_filter_v1","classifier_v1",...)
| stats avg(duration_ms), p50(duration_ms), p95(duration_ms) by name
```

### Dashboard 3: Routing Distribution

**Type:** Stacked bar chart (100% stacked)
**X-axis:** Date
**Y-axis:** % of tasks
**Series:** deterministic (green), local GGUF (blue), Fireworks API (orange), bypass (gray)
**Source:** `route_decision` from trace metadata
**Query:**
```
traces | where name == "pipeline_run"
| stats count() by decision=metadata.route_decision, _time
```

### Dashboard 4: Token Usage Over Time

**Type:** Area chart
**X-axis:** Date
**Y-axis:** Tokens (input + output)
**Series:** Area = total tokens, overlay lines per solver type
**Source:** Fireworks API token counts + local model estimates
**Query:**
```
observations | where name == "fireworks_solver_v1"
| stats sum(usage.input), sum(usage.output) by _time
```

### Dashboard 5: Eval Run Comparison Matrix

**Type:** Heatmap table
**Rows:** Eval run names
**Columns:** Metrics (classifier_accuracy, solver_accuracy, avg_latency_ms, etc.)
**Cell value:** Metric value
**Color:** Green (good) → white → red (bad)
**Source:** Score averages per eval run
**Query:**
```
scores | stats avg(value) by eval_run_name, score_name
```

### Dashboard 6: Misclassification Explorer

**Type:** Table with filters
**Columns:** task_id, prompt (truncated), original_category, corrected_category, confidence, latency_ms, timestamp
**Filters:** category, confidence range, date range
**Source:** `misclassifications` dataset
**Action:** Click row → open trace detail in Langfuse

### Creating Dashboards via SDK

```python
# agent/dashboards.py

from agent.langfuse_client import get_langfuse

def create_dashboards():
    lf = get_langfuse()

    lf.create_dashboard(
        name="Per-Category Accuracy Trends",
        widgets=[
            {
                "title": "Accuracy by Category (7d rolling)",
                "type": "line",
                "query": "...",
                "width": 12,
            },
        ]
    )

    lf.create_dashboard(
        name="Node Latency",
        widgets=[
            {"title": "Node Latency (P50/P95)", "type": "boxplot", ...},
            {"title": "Latency Drift (7d)", "type": "line", ...},
        ]
    )

    lf.create_dashboard(
        name="Routing Overview",
        widgets=[
            {"title": "Routing Distribution", "type": "stacked_bar", ...},
            {"title": "Fireworks Usage Over Time", "type": "area", ...},
        ]
    )

    lf.create_dashboard(
        name="Eval Comparison",
        widgets=[
            {"title": "Run Comparison Heatmap", "type": "heatmap", ...},
        ]
    )
```

> **Note:** Langfuse dashboard creation via SDK is available in the enterprise tier. For self-hosted (OSS), create dashboards manually via the UI → Dashboards → Create. The query patterns above are copy-paste ready for the UI query builder.

---

## 9. Implementation Checklist

### Phase 1: Foundation (Day 1)

- [ ] Add `langfuse>=2.55.0` to `requirements.txt`
- [ ] Create `agent/langfuse_client.py` (singleton)
- [ ] Add Docker Compose services (langfuse + postgres)
- [ ] Initial deploy and verify Langfuse UI at `http://localhost:3000`
- [ ] Add `LANGFUSE_ENABLE=1` env var to pipeline service

### Phase 2: Instrumentation (Day 2)

- [ ] Add `@observe` to `pre_filter.py:stage0()`
- [ ] Add `@observe` to `category_filter.py:classify_with_detail()`
- [ ] Add `@observe` to `complexity_filter.py:score()`
- [ ] Create `agent/routing.py` and add `@observe`
- [ ] Add `@observe` to deterministic solver wrapper
- [ ] Add `@observe` to `local_vote.py:solve_with_consensus()`
- [ ] Add `@observe` to `fireworks.py:FireworksSolver.solve()`
- [ ] Add `@observe` to `verify.py:verify()`
- [ ] Wrap main pipeline call in a trace
- [ ] Verify traces appear in Langfuse UI with correct span hierarchy

### Phase 3: Versioning (Day 2-3)

- [ ] Create `agent/section_versions.py`
- [ ] Set initial node version strings in all `@observe` decorators
- [ ] Add section version metadata to pipeline traces
- [ ] Test version bump (change a version, run again, verify both versions in UI)

### Phase 4: Datasets (Day 3)

- [ ] Create `agent/dataset_manager.py`
- [ ] Create `pipeline_runs` dataset via SDK
- [ ] Create `misclassifications` dataset via SDK
- [ ] Create `eval_results` dataset via SDK
- [ ] Add dataset logging to `harness.py` (after each completed task)
- [ ] Create `scripts/export_misclassifications.py`
- [ ] Test: 10 tasks → verify items in dataset

### Phase 5: Evaluations (Day 4)

- [ ] Build ground-truth dataset (annotate 50+ prompts with correct categories)
- [ ] Create classifier accuracy eval script
- [ ] Create solver accuracy eval script
- [ ] Create router decision eval script
- [ ] Create post_processor gate eval script
- [ ] Run all evals → verify scores in UI

### Phase 6: Dashboards & Comparison (Day 5)

- [ ] Create per-category accuracy trends dashboard
- [ ] Create node-level latency dashboard
- [ ] Create routing distribution dashboard
- [ ] Create token usage dashboard
- [ ] Create eval run comparison script
- [ ] Run comparison between two config variants (e.g., threshold 0.30 vs 0.10)

---

## Appendix A: Docker Compose (Complete)

```yaml
version: "3.9"

services:
  pipeline:
    build: .
    image: ghcr.io/artemkorolev1/amd-hackathon-submit
    container_name: amd-pipeline
    restart: "no"
    environment:
      # Pipeline config
      - MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
      - N_THREADS=2
      - N_CTX=2048
      - FIREWORKS_API_KEY=${FIREWORKS_API_KEY:-}
      - ALLOWED_MODELS=
      - DEADLINE_S=600
      - STAGING_ENABLED=0
      # Langfuse config
      - LANGFUSE_ENABLE=1
      - LANGFUSE_HOST=http://langfuse:3000
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-pk-local}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-sk-local}
      # Section versions (optional — defaults in code)
      - SECTION_VERSION_CLASSIFIER=classifier_group-v1
      - SECTION_VERSION_SOLVER=solver_group-v1
      - SECTION_VERSION_QUALITY=quality_group-v1
      - SECTION_VERSION_ROUTING=routing_group-v1
    depends_on:
      langfuse:
        condition: service_healthy
    volumes:
      - /input:/input:ro
      - /output:/output
      - ./models:/models:ro

  langfuse:
    image: ghcr.io/langfuse/langfuse:latest
    container_name: langfuse
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://langfuse:langfuse@postgres:5432/langfuse
      - NEXTAUTH_SECRET=${NEXTAUTH_SECRET:-change-me-in-production}
      - S3_ENABLED=false
      - REDIS_ENABLED=false
      - METRICS_DISABLED=true
      - LANGFUSE_ENABLE_EXPORT=1
      - DATABASE_MIGRATION_ENABLED=true
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "http://localhost:3000/api/public/health"]
      interval: 10s
      timeout: 5s
      retries: 10
      start_period: 30s
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    image: postgres:16-alpine
    container_name: langfuse-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=langfuse
      - POSTGRES_PASSWORD=langfuse
      - POSTGRES_DB=langfuse
    volumes:
      - langfuse-pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Weekly misclassification export (manual trigger or cron)
  misclassification-export:
    build: .
    container_name: amd-misclass-export
    profiles: ["cron"]
    entrypoint: ["python", "-u", "/scripts/export_misclassifications.py"]
    environment:
      - LANGFUSE_ENABLE=1
      - LANGFUSE_HOST=http://langfuse:3000
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-pk-local}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-sk-local}
    depends_on:
      - langfuse

volumes:
  langfuse-pgdata:
```

## Appendix B: Langfuse Resources

| Resource | URL |
|----------|-----|
| Self-hosted docs | https://langfuse.com/docs/deployment/self-hosted |
| @observe decorator | https://langfuse.com/docs/sdk/python/decorators |
| Dataset management | https://langfuse.com/docs/datasets |
| Evaluation | https://langfuse.com/docs/scores |
| Dashboard API | https://langfuse.com/docs/api/dashboards |
| Tracing overview | https://langfuse.com/docs/tracing |
