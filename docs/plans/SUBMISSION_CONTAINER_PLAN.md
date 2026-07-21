# Submission Container Verification Plan — AMD ACT II Track 1

> **Project:** Router to Vibehalla  
> **Container:** `ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>`  
> **Code repo:** `/home/artem/dev/amd-hackathon-v12h/` (or equivalent)  
> **Hackathon:** lablab.ai — AMD Developer Hackathon ACT II, Track 1  
> **Grader constraints:** 2 vCPU, 4 GB RAM, CPU-only, 10 min (600s) deadline, 30s per-task max  

---

## Table of Contents

1. [Pre-Build: Code & Architecture Verification]
2. [Dockerfile Audit]
3. [Harness Contract Verification]
4. [Build Process]
5. [Build Verification]
6. [Mode-by-Mode Smoke Tests (THE CORE)]
7. [Error State Readiness]
8. [Push & Submission]
9. [Post-Submission Monitoring]
10. [Fallback & Recovery]

---

## 1. Pre-Build: Code & Architecture Verification

Before ANY `docker build` runs, verify the code being packaged is what you intend.

### 1.1 Identify the build source

- [ ] Run `pwd` — are you in the expected working directory?
- [ ] Run `git log --oneline -1` — does the commit hash match the intended version?
- [ ] Run `git status --short` — is the working tree **clean** (no uncommitted changes)?
- [ ] Run `git rev-parse --abbrev-ref HEAD` — is this the correct branch?
- [ ] Verify no other Hermes session has switched branches in this repo recently
  ```bash
  # Check git reflog for recent checkouts
  git reflog -5 --date=iso
  ```

### 1.2 Architecture identity check

Determine which solver architecture the code implements:

- [ ] Does `agent/solvers/local_vote.py` exist? → **Local model pipeline** (llama-cpp-python GGUF)
- [ ] Does `agent/main.py` import `lora_model`? → **LoRA adapter pipeline** (transformers + PEFT)
- [ ] Is there a `harness.py` (module-level, v12h style)? → **Direct-inference harness**
- [ ] Does `agent/config.py` set `LLAMA_ENABLE = False`? → **API-only / Fireworks pipeline**
- [ ] Does the agent need `FIREWORKS_API_KEY` to function, or can it run fully offline?

**Rule:** The architecture you intend to submit must match what `docker build` will actually COPY. A `git checkout` in another terminal can swap the entire pipeline without you noticing.

### 1.3 File integrity check

- [ ] Run `git diff <last-known-good-commit> --stat` — list every file that differs
- [ ] Verify every changed file was **intentionally modified** for THIS version
- [ ] If unreviewed files are present (e.g., `classifier_ensemble.py`, `upgrade_deterministic.py`), DO NOT build from this branch — create a clean branch from last-good and cherry-pick only intended changes
- [ ] Check for **stale orphan files** that could crash the container:
  ```bash
  # Files with no .py source
  find . -name '*.pyc' -not -path './.git/*' | while read pyc; do
    py="${pyc%.pyc}.py"
    [ ! -f "$py" ] && echo "ORPHAN: $pyc (no source $py)"
  done
  # Backup/stale experiment files
  find . -name '*.bak' -o -name '*~' -o -name '*.old' | grep -v .git
  ```

### 1.4 Symlink audit (CRITICAL)

Docker `COPY` does NOT follow symlinks pointing outside the build context:

- [ ] Run: `find . -type l -exec sh -c 'readlink -f "$1" | grep -q "^$(pwd)"' _ {} \; -print`
  - If any symlink exists and its resolved path is outside the build context, that file will be **missing** at runtime → `ModuleNotFoundError`
- [ ] For every file the Dockerfile COPYs, verify it's a real file (not a broken/external symlink)
- [ ] Check `ls -la` on all top-level `.py` files — none should be symlinks to `/home/artem/dev/amd-hackathon-shared/` or similar

**Historical root cause:** A symlink `CATEGORY_REGISTRY.py -> /home/artem/dev/amd-hackathon-shared/CATEGORY_REGISTRY.py` was silently broken in the image. At runtime, `from CATEGORY_REGISTRY import get_short_name` crashed instantly.

### 1.5 Model file audit (if including a GGUF)

- [ ] Run `ls -la models/*.gguf` — check symlink status
- [ ] Run `file models/*.gguf` — verify it's "data" (not "ASCII text" = broken symlink, not "symbolic link")
- [ ] Run `du -sh models/*.gguf` — verify expected size (~900MB for 1.5B, ~2.4GB for 4B)
- [ ] If the model is downloaded during build via `curl` or `huggingface_hub`, verify the URL is valid (not a 404 redirect HTML page)

### 1.6 Build source isolation

If building from a directory with `.git`, any terminal can mutate the source mid-build:

- [ ] **Ideal:** Build from an immutable flat copy created via `git archive`
  ```bash
  mkdir -p /tmp/build-src-$(date +%s)
  git archive HEAD | tar -xC /tmp/build-src-$(date +%s)/
  docker build -t <image> /tmp/build-src-$(date +%s)/
  ```
- [ ] **Alternative:** Build from the working directory but **lock all other terminals** out
- [ ] **Never** build while another Hermes session is running `git checkout` or making changes in the same repo

---

## 2. Dockerfile Audit

### 2.1 Base image & platform

- [ ] Base image: `python:3.12-slim` (proven pattern, not ubuntu, not alpine unless fully tested)
- [ ] Platform explicitly set: `FROM python:3.12-slim` and build with `--platform linux/amd64`
- [ ] `PYTHONUNBUFFERED=1` set in ENV
- [ ] `PYTHONDONTWRITEBYTECODE=1` set (optional but good practice)
- [ ] `PIP_NO_CACHE_DIR=1` set (reduces image size)

### 2.2 Dependencies

- [ ] `requirements.txt` lists ONLY what's needed at runtime (no dev tools)
- [ ] `torch` (if needed) pinned to CPU-only index to avoid pulling CUDA (~7GB)
  ```dockerfile
  RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
  ```
- [ ] `llama-cpp-python` (if used) installed from CPU wheel index:
  ```dockerfile
  RUN pip install --no-cache-dir \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
      -r requirements.txt
  ```
- [ ] Check transitive dependency size before adding new packages:
  ```bash
  pip install --dry-run <package> 2>&1 | grep -E "^(Collecting|Installing)"
  ```
- [ ] **No secrets/API keys baked into the image** (FIREWORKS_API_KEY, tokens, etc.)
- [ ] Strip pip cache after install to save space:
  ```dockerfile
  RUN pip install ... && find /usr/local/lib -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
  ```

### 2.3 Entrypoint

- [ ] Use `ENTRYPOINT` (not `CMD`) — the grader uses ENTRYPOINT as the container's main process
- [ ] Directly to Python: `ENTRYPOINT ["python3", "-u", "-m", "agent.main"]`
- [ ] NOT a shell script wrapper (shell scripts can mask exit codes)
- [ ] The `-u` flag ensures stdout/stderr is unbuffered (critical for real-time grader logging)
- [ ] If using `harness.py` pattern: `ENTRYPOINT ["python3", "-u", "harness.py"]`
- [ ] Verify entrypoint runs without requiring arguments:
  - The grader does NOT pass additional arguments to your entrypoint
  - The entrypoint must discover input itself (/input/tasks.json, stdin, env vars)
- [ ] **Never** use `CMD` to pass args to `ENTRYPOINT` — the grader may override CMD

### 2.4 /input and /output directories

- [ ] `RUN mkdir -p /input /output` is in the Dockerfile
- [ ] Code writes to `/output/results.json` (not `/tmp/`, not `./output/`, not a custom path)
- [ ] Code reads from `/input/tasks.json` as the **primary** source (not only stdin)

### 2.5 Environment variables

Verify these env vars are set in the Dockerfile or handled in code:

| Variable | Required? | In Dockerfile? | In code fallback? |
|----------|-----------|----------------|-------------------|
| `FIREWORKS_API_KEY` | Yes (if using API) | NO (injected at runtime) | `""` → graceful fallback |
| `ALLOWED_MODELS` | Yes (grader injects) | NO | Dynamic parse |
| `DEADLINE_S` | Yes (grader injects) | NO | Default `600` |
| `TASK_COUNT` | No | NO | Default from config |
| `LLAMA_ENABLE` | If using llama.cpp | YES | Off by default |
| `MODEL_PATH` | If using local model | YES | Default path |
| `N_GPU_LAYERS` | If using local model | YES | `0` (CPU-only) |
| `N_CTX` | If using local model | YES | `2048` |
| `N_THREADS` | If using local model | YES | `2` |

### 2.6 RAM budget check

- [ ] Calculate expected RAM usage:
  - Base Python runtime: ~150MB
  - numpy/scipy/pandas: ~100MB
  - llama-cpp-python loaded: ~200-400MB
  - GGUF model file: N GB
  - KV cache + compute buffers: ~200-500MB
  - **Total must fit in 4GB** with headroom
- [ ] Known budgets from empirical testing:
  - Qwen2.5-1.5B (900MB Q4): ✅ fits comfortably
  - Phi-4-mini (2.4GB Q4): ✅ fits, borderline at 4GB limit
  - Qwen3-4B (2.4GB Q4): ✅ fits (tested: 5 tasks, 44s, no OOM under --cpus=2 --memory=4g)
  - Nemotron-3-Nano-4B (2.7GB Q4): ❌ fails (memory pressure causes TIMEOUT, not OOM)

### 2.7 Image size check

- [ ] Estimate compressed image size:
  - `python:3.12-slim` base: ~120MB compressed
  - pip deps (no torch): ~200-400MB compressed
  - torch CPU-only: ~800MB compressed
  - GGUF model: varies (900MB-2.7GB)
  - **Total compressed must be < 10GB**
- [ ] To check actual size before push:
  ```bash
  docker images <image> --format '{{.Repository}}:{{.Tag}} {{.Size}}'
  ```
  Note: Docker shows uncompressed size. GHCR reports compressed. A 6GB Docker size is typically ~2-3GB compressed.

### 2.8 Dockerfile pitfalls checklist

- [ ] `apt-get` packages: only what's needed for runtime, no build-essential retained
- [ ] If `build-essential` or `cmake` installed for compilation, they MUST be purged:
  ```dockerfile
  RUN apt-get purge -y build-essential cmake && apt-get autoremove -y
  ```
- [ ] `rm -rf /var/lib/apt/lists/*` after apt installs
- [ ] Model download uses `curl -L --retry 3` (not bare `curl`) or `huggingface_hub` with retry
- [ ] No `COPY . .` at the end (copies Dockerfile, .git, etc.) — explicit COPY for each directory
- [ ] Dockerfile line count: every line should be intentional. Comment or remove dead code

---

## 3. Harness Contract Verification

### 3.1 Input contract

The grader mounts tasks at `/input/tasks.json:ro`. Format:

```json
[
  {"task_id": "t1", "prompt": "What is the capital of France?"},
  {"task_id": "t2", "prompt": "Write a Python function to check if a string is a palindrome."}
]
```

- [ ] Code reads `/input/tasks.json` FIRST (before falling back to stdin)
- [ ] Code handles both string-only arrays `["task1", "task2"]` and object arrays `[{"task_id":"t1","prompt":"..."}]`
- [ ] Code handles `TASK_COUNT` env var to limit number of tasks
- [ ] Code handles `DEADLINE_S` env var — wraps in `try/except ValueError, TypeError`:
  ```python
  try:
      deadline = time.monotonic() + int(os.environ.get("DEADLINE_S", str(MAX_RUNTIME_SEC)))
  except (ValueError, TypeError):
      logger.warning("DEADLINE_S invalid, using default %ds", MAX_RUNTIME_SEC)
      deadline = time.monotonic() + MAX_RUNTIME_SEC
  ```

### 3.2 Output contract

- [ ] Writes `/output/results.json` (exact path, not `/tmp/`, not `./output/`)
- [ ] Format: `[{"task_id": "t1", "answer": "Paris"}, {"task_id": "t2", "answer": "..."}]`
- [ ] One entry per input task, in order
- [ ] Uses atomic write pattern (write to `.tmp`, then `os.replace`):
  ```python
  tmp = "/output/results.json.tmp"
  with open(tmp, "w") as f:
      json.dump(results, f)
  os.replace(tmp, "/output/results.json")
  ```
- [ ] Flushes after every task or on partial completion — NOT only at the end
- [ ] Stdout: prints one answer per line (fallback for pipe-based usage)
- [ ] Stderr: all logging/status output

### 3.3 Runtime constraints

| Constraint | Value | Check |
|------------|-------|-------|
| Tasks count | Up to 19 (confirmed) | Handle variable count via TASK_COUNT |
| Per-task max | 30 seconds | Timeout enforcement per question |
| Total deadline | 600 seconds (10 min) | DEADLINE_S + internal hard deadline |
| Startup | < 60 seconds | Model must load fast |
| CPU | 2 vCPU | `n_threads=2` |
| RAM | 4 GB | Test under `--memory=4g` |
| Swap | None | Test under `--memory-swap=4g` (= no swap) |
| Architecture | linux/amd64 | Build with `--platform linux/amd64` |
| Compressed size | < 10 GB | Check compressed size on GHCR |

### 3.4 Graceful degradation for missing env vars

- [ ] No `FIREWORKS_API_KEY` → local inference only, run without Fireworks
- [ ] No `ALLOWED_MODELS` → use model config defaults
- [ ] No `DEADLINE_S` → use default deadline (MAX_RUNTIME_SEC)
- [ ] No `/input/tasks.json` → fall back to stdin (for flexible usage)
- [ ] No tasks on stdin → exit gracefully (log warning, write empty output)

### 3.5 Error handling for grader failures

- [ ] Fireworks API 401/403 → log warning, fall back to local model
- [ ] Fireworks API 429 → retry with exponential backoff (1s, 2s, 4s), then fall back
- [ ] Fireworks API 500-504 → retry with backoff, then fall back
- [ ] Circuit breaker pattern: after 5 consecutive API failures, cool off for 30s
- [ ] Local model OOM → catch gracefully, skip to next task
- [ ] Local model timeout → return empty string, continue to next task
- [ ] Deadline approaching → reduce quality (fewer consensus samples, simpler prompts)
- [ ] **Never crash the whole container** — every exception is caught per-task, empty answer is better than RUNTIME_ERROR

---

## 4. Build Process

### 4.1 Pre-build checklist

- [ ] Git working tree is clean (`git status` = nothing to commit)
- [ ] Use `git archive` or verified flat-copy directory (see section 1.6)
- [ ] All source files exist and are not symlinks (see section 1.4)
- [ ] Model file is a real GGUF (see section 1.5)
- [ ] `git diff --stat` against last-known-good reviewed

### 4.2 Build command

```bash
docker buildx build \
  --platform linux/amd64 \
  -t ghcr.io/artemkorolev1/amd-hackathon-submit:<tag> \
  --load \
  <build-context-dir>
```

- [ ] `--platform linux/amd64` is mandatory (grader runs amd64)
- [ ] `--load` makes the image available locally for testing
- [ ] Tag is descriptive (e.g., `v12h-phi4`, not just `latest`)
- [ ] Build completes without error (exit code 0)
- [ ] Check image created: `docker images ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>`

### 4.3 Post-build verification

- [ ] Image exists locally: `docker image inspect ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>`
- [ ] Image platform is correct: `docker inspect <image> --format '{{.Os}}/{{.Architecture}}'` → `linux/amd64`
- [ ] Image size recorded: `docker images <image> --format '{{.Size}}'`
- [ ] ENTRYPOINT recorded: `docker inspect <image> --format '{{json .Config.Entrypoint}}'`

---

## 5. Build Verification (Compare Against Reference)

### 5.1 Extract and diff against last-known-good

If you have a last-known-good image (e.g., `v6.1` that scored 84.2%):

```bash
# Create containers
docker create --name ref ghcr.io/artemkorolev1/amd-hackathon-submit:v6.1
docker create --name new ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>

# Export and compare
docker cp ref:/agent/. /tmp/ref-agent/
docker cp new:/agent/. /tmp/new-agent/
diff -rq /tmp/ref-agent/ /tmp/new-agent/

# Clean up
docker rm ref new
```

- [ ] Differences are ONLY the intentional changes
- [ ] No unexpected files in the new image
- [ ] No unexpected files MISSING from the new image

### 5.2 Quick import sanity check

```bash
docker run --rm --entrypoint python3 <image> -c "
import sys
sys.path.insert(0, '/')
from agent.main import main as agent_main
print('agent.main imports OK')
from agent.stage0 import stage0
print('stage0 imports OK')
from agent.stage2 import classify
print('stage2 imports OK')
from agent.config import *
print('config imports OK')
"
```

- [ ] All imports succeed (no ModuleNotFoundError, NameError, AttributeError)

---

## 6. Mode-by-Mode Smoke Tests (THE CORE)

Four distinct test modes — ALL must pass before pushing. Do not skip any mode.

### 6.1 Mode A: No API + Grader Constraints (Tests startup, imports, RAM budget)

Purpose: Verifies the container doesn't crash on startup, can read input, produces correct output format, and doesn't OOM under exact grader resource limits.

```bash
# Create test tasks (2 diverse tasks across different categories)
cat > /tmp/test_a.json << 'ENDJSON'
[
  {"task_id":"t1","prompt":"What is 15% of 240?"},
  {"task_id":"t2","prompt":"Extract all named entities from: Dr. Sarah Johnson from Stanford University presented at the conference in New York on January 15, 2025."}
]
ENDJSON
mkdir -p /tmp/test_a_out

# Run with strict grader constraints: 2 vCPU, 4 GB RAM, NO SWAP
docker run --rm \
  --cpus=2 --memory=4g --memory-swap=4g \
  -v /tmp/test_a.json:/input/tasks.json:ro \
  -v /tmp/test_a_out:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>
```

**Verification checklist:**
- [ ] Exit code 0 (not 137 = OOM, not 1 = crash)
- [ ] Container completes in < 30s total
- [ ] `/tmp/test_a_out/results.json` exists
- [ ] File contains valid JSON
- [ ] File has exactly 2 entries (one per task)
- [ ] Each entry has `task_id` and `answer` fields
- [ ] Answers are non-empty
- [ ] No `[ERROR]`, `Traceback`, `TypeError`, `NameError`, `ModuleNotFoundError` in stderr

### 6.2 Mode B: With Grader Env Vars + Mock API Key (Tests Fireworks init, config parsing, error handling)

Purpose: Verifies all env-var-gated code paths. The grader sets `FIREWORKS_API_KEY`, `ALLOWED_MODELS`, `DEADLINE_S`. A mock API key should trigger graceful fallback (not crash).

```bash
cat > /tmp/test_b.json << 'ENDJSON'
[
  {"task_id":"t1","prompt":"What is the capital of France?"},
  {"task_id":"t2","prompt":"Write a Python function to check if a string is a palindrome."}
]
ENDJSON
mkdir -p /tmp/test_b_out

# Run with grader env vars + mock API key
docker run --rm \
  --cpus=2 --memory=4g --memory-swap=4g \
  -e FIREWORKS_API_KEY=mock_test_key_123 \
  -e ALLOWED_MODELS="accounts/fireworks/models/deepseek-v4-flash" \
  -e DEADLINE_S=600 \
  -v /tmp/test_b.json:/input/tasks.json:ro \
  -v /tmp/test_b_out:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:<tag> 2>&1
```

**Verification checklist:**
- [ ] Exit code 0
- [ ] Logs show "Fireworks solver ready" or "FIREWORKS_API_KEY" detected
- [ ] NO `TypeError: ... missing 1 required positional argument`
- [ ] NO `NameError`, `AttributeError`, `ModuleNotFoundError`
- [ ] ALLOWED_MODELS parsed correctly (stderr shows the model name)
- [ ] DEADLINE_S parsed correctly (stderr shows the deadline value)
- [ ] All tasks produce non-empty answers (fallback because mock key → 401)
- [ ] Circuit breaker does NOT trip (mock key 401s should be isolated per-task failures)

**CRITICAL checks this mode catches:**
- `breaker.record_failure()` called without required `error_type` argument
- `parse_allowed_models()` called with wrong signature
- `DEADLINE_S` int-cast crash (if set to non-integer string by grader)
- Fireworks init code that crashes when API key is present but invalid

### 6.3 Mode C: Full Pipeline Coverage — 8+ Diverse Tasks (Exercises ALL solver paths)

Purpose: Tests every category, every solver path (deterministic, local model, Fireworks escalation), and the classifier routing.

```bash
cat > /tmp/test_c.json << 'ENDJSON'
[
  {"task_id":"t1","prompt":"What is the capital of France?"},
  {"task_id":"t2","prompt":"If a train travels 120 km in 2 hours, what is its speed in km/h?"},
  {"task_id":"t3","prompt":"Extract all named entities from: On January 15, 2025, Dr. Sarah Johnson from Stanford University and Prof. Michael Chen from MIT published a breakthrough paper on quantum computing in Nature."},
  {"task_id":"t4","prompt":"Write a Python function called is_palindrome that takes a string and returns True if it's a palindrome, False otherwise."},
  {"task_id":"t5","prompt":"Classify the sentiment of this text: The movie was absolutely terrible. The acting was wooden, the plot was predictable, and the special effects looked like they were from 1995. I want my money back."},
  {"task_id":"t6","prompt":"Solve: All mathematicians are logical. Some programmers are mathematicians. Can we conclude that some programmers are logical?"},
  {"task_id":"t7","prompt":"Summarize: Artificial intelligence has transformed numerous industries over the past decade. From healthcare diagnostics to autonomous vehicles, machine learning algorithms are increasingly making decisions that were once the exclusive domain of human experts. However, concerns about bias, transparency, and accountability remain significant challenges."},
  {"task_id":"t8","prompt":"Fix the bug in this Python function:\n\ndef find_max(lst):\n    max_val = 0\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val\n\nThe function returns 0 for lists with all negative numbers."},
  {"task_id":"t9","prompt":"What is the chemical formula for water?"},
  {"task_id":"t10","prompt":"If it takes 10 minutes to boil 1 egg, how many minutes does it take to boil 10 eggs?"}
]
ENDJSON
mkdir -p /tmp/test_c_out

docker run --rm \
  --cpus=2 --memory=4g --memory-swap=4g \
  -v /tmp/test_c.json:/input/tasks.json:ro \
  -v /tmp/test_c_out:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:<tag> 2>&1
```

**Verification checklist:**
- [ ] Exit code 0
- [ ] `/tmp/test_c_out/results.json` has exactly 10 entries
- [ ] Every entry has a non-empty answer
- [ ] No task exceeded ~30s (check stderr timestamps for gaps > 35s)
- [ ] Total runtime under 300s (ok for 10 tasks, grader allows 600s for ~19)
- [ ] All category-specific classifier log lines appear (S2 classification for each)
- [ ] Deterministic solvers fire for math/NER/sentiment tasks where applicable
- [ ] Local model fires for complex tasks (code_gen, complex logic)
- [ ] Answers are format-correct (sentiment = single word, NER = categorized, code = in fences)

### 6.4 Mode D: Real Fireworks API Key (Tests end-to-end solver path)

Purpose: Verifies Fireworks model resolution, base URL construction, response parsing, and the full chain works with actual API calls.

```bash
cat > /tmp/test_d.json << 'ENDJSON'
[
  {"task_id":"t1","prompt":"What is the capital of France?"},
  {"task_id":"t2","prompt":"Explain quantum entanglement in one sentence."}
]
ENDJSON
mkdir -p /tmp/test_d_out

docker run --rm \
  --cpus=2 --memory=4g --memory-swap=4g \
  -e FIREWORKS_API_KEY="$(cat ~/.fireworks_api_key 2>/dev/null || echo "$FIREWORKS_API_KEY")" \
  -e ALLOWED_MODELS="accounts/fireworks/models/deepseek-v4-flash" \
  -e DEADLINE_S=600 \
  -v /tmp/test_d.json:/input/tasks.json:ro \
  -v /tmp/test_d_out:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:<tag> 2>&1
```

**Verification checklist:**
- [ ] Exit code 0
- [ ] Logs show "Fireworks solver ready" (not "No FIREWORKS_API_KEY")
- [ ] Logs show Fireworks being called (model name + response in stderr)
- [ ] Answers are correct (Paris, etc.)
- [ ] No HTTP errors (401, 403 = bad key; 429 = rate limited)
- [ ] Total runtime < 30s (Fireworks calls are fast, <5s each)
- [ ] Base URL is correct (no double `/v1`, no missing `/chat/completions`)

### 6.5 Mode Summary Table

| Mode | Tests | Time | API Key | Resource Limit | Priority |
|------|-------|------|---------|----------------|----------|
| A | Startup, imports, RAM, output format | ~10s | None | `--cpus=2 --memory=4g` | **Must pass first** |
| B | Env var handling, error paths | ~10s | Mock | `--cpus=2 --memory=4g` | **Must pass second** |
| C | Full pipeline, all categories | ~60-120s | None | `--cpus=2 --memory=4g` | **Must pass third** |
| D | Real API, end-to-end | ~15s | Real | `--cpus=2 --memory=4g` | **Must pass fourth** |

---

## 7. Error State Readiness

### 7.1 Understanding grader error states

The grader reports these after submission:

| Error | Root Cause | How to Prevent | How to Fix |
|-------|-----------|----------------|------------|
| `PULL_ERROR` | Image private, wrong tag, no arm64 manifest, or re-saved before tag existed | Make GHCR package PUBLIC. Tag correctly. Check `docker manifest inspect`. Push BEFORE re-saving form. | Fix visibility/tag and re-save |
| `RUNTIME_ERROR` | Uncaught exception. Import error, TypeError, OOM, etc. | Mode A, B, C tests above all pass. Test under `--cpus=2 --memory=4g`. | Smash the exception with try/except in main loop. Add `logging.exception()` everywhere. |
| `INFRA_ERROR` | Transient grader harness failure (image was fine). | Re-save the same image (don't rebuild). | Wait and re-save |
| `TIMEOUT` | Container ran past deadline (600s) | Check per-task timing in Mode C. Total < 600s. Reduce consensus samples. | Tighten time budget ratchets. Cap Fireworks calls. |
| `OUTPUT_MISSING` | Crashed before writing /output/results.json | Write in `finally` block. Flush after every N tasks. | Add crash-proof writes |
| `INVALID_RESULTS_SCHEMA` | Output doesn't match spec `[{"task_id","answer"}]` | Validate output in tests. No extra fields. | Fix format |
| `MODEL_VIOLATION` | Called a non-allowed model | Read ALLOWED_MODELS at runtime. Validate before calling. | Add model whitelist check |
| `IMAGE_TOO_LARGE` | Compressed > 10GB | Check compressed size. Strip pip cache, build deps. | Remove unnecessary deps |
| `ACCURACY_GATE_FAILED` | Answer quality below 84.2% threshold | Run eval against ground truth. Improve prompts, routing, model. | Improve pipeline |

### 7.2 Must-have protections against every error state

- [ ] **RUNTIME_ERROR prevention:** Every task handler is wrapped in `try/except Exception` — single-task failure doesn't crash the container
- [ ] **OUTPUT_MISSING prevention:** `results.json` is written in a `finally` block or at interrupt signal
- [ ] **TIMEOUT prevention:** Per-question timeout (thread with `future.result(timeout=28)`) — hard deadline per task
- [ ] **MODEL_VIOLATION prevention:** Read ALLOWED_MODELS from env, only use models from that list
- [ ] **INVALID_RESULTS_SCHEMA prevention:** Output is validated JSON before write
- [ ] **PULL_ERROR prevention:** Verify `docker logout ghcr.io && docker pull <image>` works before submitting

### 7.3 Signal handling

- [ ] `SIGTERM` caught to flush output before exit
- [ ] `SIGINT` caught for local testing
- [ ] On any signal, `/output/results.json` is written with whatever results exist

```python
import signal

def _flush_on_exit(signum, frame):
    logger.warning("Received signal %d — flushing results", signum)
    _flush(answers)
    sys.exit(0)

signal.signal(signal.SIGTERM, _flush_on_exit)
signal.signal(signal.SIGINT, _flush_on_exit)
```

---

## 8. Push & Submission

### 8.1 Push checklist

- [ ] **User has explicitly approved the push** (never push without confirmation on first push of the session)
- [ ] All 4 test modes (A, B, C, D) pass
- [ ] Image exists locally with the correct tag

```bash
# Push to GHCR
docker push ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>
```

- [ ] Push succeeds (exit 0, all layers uploaded)
- [ ] Record the image digest: `docker inspect <image> --format '{{.RepoDigests}}'`

### 8.2 Post-push verification

```bash
# 1. Verify push completed (check GHCR API)
curl -s "https://ghcr.io/v2/artemkorolev1/amd-hackathon-submit/manifests/<tag>" | head -1
# Should return a valid JSON, not 404

# 2. Test pull without auth (simulates grader)
docker logout ghcr.io
docker pull ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>
```

- [ ] `docker pull` works without auth (grader pulls anonymously)
- [ ] Digest matches what was pushed
- [ ] GHCR package is PUBLIC:
  - Go to `https://github.com/users/artemkorolev1/packages/container/amd-hackathon-submit/settings`
  - Make sure "Package visibility" is PUBLIC (under "Danger Zone")
  - Or verify via API: `curl -sI "https://ghcr.io/v2/artemkorolev1/amd-hackathon-submit/manifests/<tag>" | grep -i "content-type"`

### 8.3 GHCR package visibility fix (if needed)

If the package is private:

```bash
# Option 1: GitHub API (requires PAT with delete:packages, write:packages, repo scopes)
curl -X PATCH \
  -H "Authorization: Bearer $(cat ~/.ghcr-token)" \
  -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  "https://api.github.com/user/packages/container/amd-hackathon-submit/visibility" \
  -d '{"visibility": "public"}'

# Option 2: GitHub UI
# 1. Go to https://github.com/users/artemkorolev1/packages/container/amd-hackathon-submit
# 2. Click "Package settings" (gear icon)
# 3. Scroll to "Danger Zone" → "Change visibility" → "Make public"
```

### 8.4 GHCR auth for push

```bash
# Use token saved in ~/.ghcr-token (has write:packages scope)
cat ~/.ghcr-token | docker login ghcr.io -u artemkorolev1 --password-stdin
# Note: gh auth token lacks write:packages scope — use ~/.ghcr-token explicitly
```

### 8.5 Submission form

1. Go to lablab.ai → your AMD ACT II project
2. Locate the submission form
3. Update the Docker image reference: `ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>`
   - Use the **pinned version tag** (`:v12h`), NOT `:latest`
4. Click "Update Submission" or "Save"

- [ ] Image reference uses the pinned tag (not `:latest`) — versioned tags never get overwritten
- [ ] Re-save is the ONLY action that triggers a new grader check
- [ ] Re-save was done AFTER the push completed (order matters!)
- [ ] GitHub repo URL on the form matches the intended repo

**Pitfall:** Re-saving BEFORE the tag exists on GHCR guarantees PULL_ERROR. Sequence: push → verify → re-save.

**Pitfall:** Re-saving binds the tag at click time, not push time. If you push v2 at 22:00, push v3 at 22:15, then re-save at 22:30 — only v3 gets evaluated.

### 8.6 Tag both :tag and :latest

```bash
docker tag ghcr.io/artemkorolev1/amd-hackathon-submit:<tag> ghcr.io/artemkorolev1/amd-hackathon-submit:latest
docker push ghcr.io/artemkorolev1/amd-hackathon-submit:latest
```

But use the pinned tag in the submission form.

### 8.7 Record everything

After push and before moving on:

- [ ] Record in VERSION_LOCKED.md (create if doesn't exist):
  - Version tag (e.g., `v12h-phi4`)
  - Docker image: `ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>`
  - Digest: `sha256:...`
  - Git commit: full hash
  - Summary: what changed, model used
  - Date/time of push

- [ ] Update the project's CONTEXT.md:
  - Add row to `## Versions > Submission Log` table
  - Update `## Status` section
  - Update `Current GHCR tags` block

---

## 9. Post-Submission Monitoring

### 9.1 Monitoring checklist

After the form is re-saved, the grader enters a queue. Check results periodically:

- [ ] Set up a cron job (or manual check every 30-60 min):
  ```bash
  # Check GHCR pull counter (increments = grader pulled your image)
  curl -s "https://ghcr.io/v2/artemkorolev1/amd-hackathon-submit/manifests/<tag>" | jq '.'
  
  # Check lablab leaderboard
  curl -s "https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/live" | grep -i "router\|vibehalla\|artem"
  ```

- [ ] Check `ghcr.io` package download count — if it increased, the grader pulled your image
- [ ] Check the lablab.ai project page for status updates
- [ ] Check the leaderboard for accuracy/token scores

### 9.2 Grader state timeline

From empirical observation:
- **Typical queue time:** 15 min to 6 hours (depends on queue depth)
- **Processing cadence:** ~28-56 events per hour
- **Error re-check:** Fix an error, re-save → goes to back of queue → re-checked in ~1-2 hours
- **INFRA_ERROR** is transient — re-saving the same image usually clears it on next check
- **RUNTIME_ERROR** after a PULL_ERROR fix is common — the grader tests different things in sequence

### 9.3 Result interpretation

```text
RUNTIME_ERROR
Your container was pulled but crashed during evaluation.
Check your entrypoint/CMD and that it runs on linux/amd64.
```
→ The image pulled successfully but the ENTRYPOINT/CMD failed. See section 7.

```text
INFRA_ERROR
This submission could not be scored.
Review your submission and re-save to try again.
```
→ Grader infrastructure glitch. Re-save ONLY (no rebuild needed). The image is fine.

```text
ACCURACY_GATE_FAILED
Your submission scored X% (requires 84.2% to qualify).
```
→ Container ran fine, answers were below threshold. Improve prompts/routing/model.

```text
PULL_ERROR
```
→ Image not found, not public, or wrong platform. Check tag, visibility, and `linux/amd64`.

```text
TIMEOUT
```
→ Container took > 600s. Tighten time budget.

---

## 10. Fallback & Recovery

### 10.1 If Mode A fails (crash under constraints)

- [ ] Check if it's OOM (exit code 137) — reduce model size or set `LLAMA_ENABLE=0`
- [ ] Check for import errors — run import sanity check (section 5.2)
- [ ] Check for symlink issues (section 1.4)
- [ ] Run with more verbose logging: add `-e DEBUG=1` or modify logging level
- [ ] Run without resource constraints to confirm the code works at all:
  ```bash
  docker run --rm -v /tmp/test.json:/input/tasks.json:ro <image>
  ```

### 10.2 If Mode B fails (env-var gated crash)

- [ ] Grep stderr for `TypeError`, `NameError`, `AttributeError`
- [ ] Check `breaker.record_failure()` — requires `error_type` string argument
- [ ] Check `DEADLINE_S` parsing — `int(os.environ.get(...))` can crash on bad values
- [ ] Check `ALLOWED_MODELS` parsing — empty string, malformed list
- [ ] Check Fireworks init path — `__init__` calling `urllib.error` before checking API key

### 10.3 If Mode C fails (wrong answers or format)

- [ ] Check each answer individually — what did the model produce vs expected?
- [ ] Check if deterministic solvers are returning wrong answers for tasks they shouldn't handle
- [ ] Check if the local model is hallucinating vs Fireworks giving better answers
- [ ] Check output format — is `results.json` valid? Does each entry have `task_id` and `answer`?
- [ ] Check if some tasks timed out (empty answers)
- [ ] Run a single-task test for the category that failed

### 10.4 If Mode D fails (Fireworks API)

- [ ] Check API key is valid (not expired, not revoked)
- [ ] Check model name is valid and accessible from the key's account
- [ ] Check Fireworks base URL — `https://api.fireworks.ai/inference/v1` (not `.../v1/`)
- [ ] Check for HTTP 429 (rate limit) — reduce request rate
- [ ] Check for HTTP 401/403 — bad key or model not in account

### 10.5 If PULL_ERROR after push

- [ ] Run `docker logout ghcr.io && docker pull ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>` — if this fails, image is private or tag doesn't exist
- [ ] Check GHCR package visibility (must be PUBLIC)
- [ ] Re-run push (maybe push failed silently)
- [ ] Check manifest: `docker manifest inspect ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>`
- [ ] Re-save the form AFTER the tag is confirmed public and pullable

### 10.6 If INFRA_ERROR

- [ ] Do NOT rebuild or push a new image
- [ ] Simply re-save the same form on lablab.ai (same tag, no changes)
- [ ] Wait for next check cycle

### 10.7 If RUNTIME_ERROR after push (but Mode A/B/C/D passed locally)

- [ ] The grader environment may differ from your local Docker:
  - Different CPU architecture flags (AVX vs no AVX)
  - Different kernel version
  - Different Docker version (OCI attestation rejection)
  - Grader pulls the tag → if tag was `:latest`, it might have been overwritten
- [ ] Check: was the image tag the same on the lablab form as what you tested?
- [ ] Build with `--provenance=false` to avoid OCI attestation issues:
  ```bash
  docker buildx build --provenance=false --platform linux/amd64 -t <image> .
  ```
- [ ] Set `n_threads=2` explicitly for grader's 2 vCPU

---

## Quick Reference: One-Line Commands

### Pre-build
```bash
git status --short && git log --oneline -1 && git rev-parse --abbrev-ref HEAD
find . -type l -not -path './.git/*'
find . -name '*.pyc' -not -path './.git/*' | while read pyc; do [ ! -f "${pyc%.pyc}.py" ] && echo "ORPHAN: $pyc"; done
```

### Build
```bash
docker buildx build --platform linux/amd64 -t ghcr.io/artemkorolev1/amd-hackathon-submit:<tag> --load .
```

### All 4 test modes (copy-paste block)
```bash
# Mode A
docker run --rm --cpus=2 --memory=4g --memory-swap=4g -v /tmp/test_a.json:/input/tasks.json:ro -v /tmp/test_a_out:/output <image>; echo "Exit: $?"

# Mode B
docker run --rm --cpus=2 --memory=4g --memory-swap=4g -e FIREWORKS_API_KEY=mock_test_key_123 -e ALLOWED_MODELS="accounts/fireworks/models/deepseek-v4-flash" -e DEADLINE_S=600 -v /tmp/test_b.json:/input/tasks.json:ro -v /tmp/test_b_out:/output <image> 2>&1 | grep -E "Fireworks|TypeError|error|traceback|breaker"

# Mode C
docker run --rm --cpus=2 --memory=4g --memory-swap=4g -v /tmp/test_c.json:/input/tasks.json:ro -v /tmp/test_c_out:/output <image> 2>&1 | tail -30

# Mode D
docker run --rm --cpus=2 --memory=4g --memory-swap=4g -e FIREWORKS_API_KEY="$(cat ~/.fireworks_api_key 2>/dev/null || echo $FIREWORKS_API_KEY)" -e ALLOWED_MODELS="accounts/fireworks/models/deepseek-v4-flash" -e DEADLINE_S=600 -v /tmp/test_d.json:/input/tasks.json:ro -v /tmp/test_d_out:/output <image> 2>&1 | grep -E "Fireworks|answer|error|HTTP"
```

### Push
```bash
cat ~/.ghcr-token | docker login ghcr.io -u artemkorolev1 --password-stdin
docker push ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>
docker push ghcr.io/artemkorolev1/amd-hackathon-submit:latest
```

### Verify after push
```bash
docker logout ghcr.io
docker pull ghcr.io/artemkorolev1/amd-hackathon-submit:<tag>
curl -s "https://ghcr.io/v2/artemkorolev1/amd-hackathon-submit/manifests/<tag>" | head -5
```

---

*Generated from the hackathon-submit-verify skill, empirical testing of 10+ submission cycles, and root-cause analysis of every error state encountered (PULL_ERROR, RUNTIME_ERROR, INFRA_ERROR, ACCURACY_GATE_FAILED, TIMEOUT).*
