# Build & Test Plan — AMD ACT II Track 1 Submission Container

**Based on official specs from:** `AMD Hackathon Judging FAQ and Self-Check Guide.docx`  
**Code:** `/home/artem/dev/amd-hackathon/` | **Entrypoint:** `harness.py` | **Model:** Qwen2.5-1.5B GGUF

---

## Phase 0 — Pre-Flight Checks

### 0.1 Verify current state
```bash
cd ~/dev/amd-hackathon
git status --short          # Must be clean or known-dirty
git log --oneline -1        # Confirm the commit you're building from
git rev-parse --abbrev-ref HEAD
```

### 0.2 Know your architecture
The codebase has TWO pipeline implementations:
- `harness.py` (622 lines) — **THIS is the entrypoint** (`ENTRYPOINT ["python3", "-u", "harness.py"]`)
- `agent/main.py` (384 lines) — NOT used in container, lives in image but never called

**Everything below operates on `harness.py` only.** Do not touch `agent/main.py` logic.

---

## Phase 1 — Fix the 2 Critical Bugs (NO logic changes)

These are the only changes to the code. No pipeline logic, no routing, no prompts, no model — just fixing broken references.

### 1.1 Fix imports in `harness.py` (line 71-72)

**Bug:** `from agent.stage0 import stage0` and `from agent.stage2 import classify_with_detail` — those modules don't exist.

**Fix:** Replace with the actual module names:
```python
from agent.pre_filter import stage0
from agent.category_filter import classify_with_detail as _stage2_detail
```

**Verify:** `python3 -c "from agent.pre_filter import stage0; from agent.category_filter import classify_with_detail as _stage2_detail; print('OK')"` exits 0.

### 1.2 Fix task input path in `harness.py` (lines 601-606)

**Bug:** Reads `sys.argv[1]` or non-existent `eval_mini_10.json` instead of grader's `/input/tasks.json`.

**Fix:** Replace the `__main__` block with a `_read_tasks()` function:
- Check `/input/tasks.json` first (the grader's mount)
- Fall back to `sys.argv[1]` if provided (for local testing)
- Fall back to stdin if neither exists (for CLI piping)

The pattern already exists in `agent/main.py` lines 93-125 — port that approach (not the logic, just the I/O part).

**Verify:** Create a test file at `/tmp/test.json`, mount it as `/input/tasks.json`, run the container — it reads tasks.

### 1.3 What NOT to change

| Do NOT touch | Reason |
|-------------|--------|
| Any `_process()` logic (lines 356-589) | Pure pipeline routing — leave as-is |
| Deterministic solvers | Accuracy tuning, not build fixes |
| System prompts or category maps | Accuracy tuning, not build fixes |
| `agent/main.py` or `agent/pipeline.py` | Dead code in container, leave as-is |
| `agent/config.py` values | Already fine for 1.5B + 2 vCPU |

---

## Phase 2 — Build the Container

### 2.1 Verify files to be packaged
```bash
# Check all imports in harness.py resolve
python3 -c "
from agent.pre_filter import stage0
from agent.category_filter import classify_with_detail as _stage2_detail
from agent.complexity import score as mlm_complexity
from agent.solvers.deterministic import solve_arithmetic, solve_logic, solve_sentiment, solve_ner, solve_factual_qa, solve_code_debugging
from agent.solvers.fireworks import FireworksSolver
from agent.solvers.fw_router import route as _fw_route
from agent.dynamic_prompts import build_system_prompt, build_merged_prompt, get_max_tokens as dp_max_tokens, get_stop_sequences as dp_stop_sequences
print('All imports OK')
"
```

### 2.2 Create .dockerignore
```bash
cat > .dockerignore << 'EOF'
.git/
.gitignore
__pycache__/
*.pyc
*.bak
upgrade_*.py
*.md
*.docx
.venv/
venv/
json
eval_results/
reports/
input/
docs/
presentation/
research/
scripts/
config/
data/
archive/
archived/
tests/
.DS_Store
Makefile
LICENSE
README.md
CONTEXT.md
HANDOFF.md
run_counter.json
EOF
```

### 2.3 Verify model file
```bash
ls -la models/*.gguf                    # Must exist, not a symlink
file models/*.gguf                      # Must say "data", not "symbolic link" or "ASCII text"
du -sh models/*.gguf                    # Expected: ~1.1GB for Qwen2.5-1.5B Q4
```

### 2.4 Build
```bash
docker buildx build \
  --platform linux/amd64 \
  -t ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix \
  --load .
```

### 2.5 Verify the image
```bash
# Size check
docker images ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix --format "{{.Size}}"

# Platform check
docker inspect ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix \
  --format '{{.Os}}/{{.Architecture}}'   # Must show linux/amd64

# Entrypoint check
docker inspect ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix \
  --format '{{json .Config.Entrypoint}}'  # Must show ["python3", "-u", "harness.py"]
```

---

## Phase 3 — Test Against Official Specs

The official FAQ says, before submitting:
> *"Pull your exact Docker image tag from a clean machine. Run the container without local files or manual setup. Confirm it writes the expected output. Validate the JSON output. Return results for every required task. Stay under the runtime limit. Make sure your repo or image tag is public and accessible."*

### 3.1 Mode A — Startup + Imports + RAM (no API key)

Tests the official "Container runs cleanly" and "All dependencies are inside the image" checks.

```bash
cat > /tmp/test_a.json << 'ENDJSON'
[
  {"task_id":"t1","prompt":"What is 15% of 240?"},
  {"task_id":"t2","prompt":"Extract all named entities from: On March 15 2023, Sundar Pichai announced that Google would open a new AI research lab in Zurich, partnering with ETH Zurich."},
  {"task_id":"t3","prompt":"Classify the sentiment: The product arrived two days late and the packaging was damaged, but the item worked perfectly."}
]
ENDJSON
mkdir -p /tmp/test_a_out

docker run --rm \
  --cpus=2 --memory=4g --memory-swap=4g \
  -v /tmp/test_a.json:/input/tasks.json:ro \
  -v /tmp/test_a_out:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix
echo "Exit code: $?"
```

**Pass criteria:**
- [ ] Exit code 0 (NOT 137 = OOM, NOT 1 = crash)
- [ ] `/tmp/test_a_out/results.json` exists
- [ ] File is valid JSON with 3 entries
- [ ] Each entry has `task_id` + `answer`
- [ ] No `ModuleNotFoundError`, `NameError`, `TypeError`, `AttributeError`, `FileNotFoundError`
- [ ] Container completes in < 30 seconds (3 tasks)
- [ ] Answers are non-empty

### 3.2 Mode B — With grader env vars + mock API key

Tests the official "No private secrets are required" + env var handling checks.

```bash
cat > /tmp/test_b.json << 'ENDJSON'
[
  {"task_id":"t1","prompt":"Name the three primary colors in the RGB color model."},
  {"task_id":"t2","prompt":"A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?"}
]
ENDJSON
mkdir -p /tmp/test_b_out

docker run --rm \
  --cpus=2 --memory=4g --memory-swap=4g \
  -e FIREWORKS_API_KEY=mock_test_key_12345 \
  -e ALLOWED_MODELS="accounts/fireworks/models/deepseek-v4-flash" \
  -e DEADLINE_S=600 \
  -v /tmp/test_b.json:/input/tasks.json:ro \
  -v /tmp/test_b_out:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix 2>&1 | grep -E "Fireworks|TypeError|error|traceback|breaker|DEADLINE"
echo "Exit code: $?"
```

**Pass criteria:**
- [ ] Exit code 0
- [ ] NO `TypeError: ... missing 1 required positional argument` (breaker.record_failure bug)
- [ ] NO `ValueError` (DEADLINE_S int-cast bug)
- [ ] All tasks produce non-empty answers (fallback because mock key → Fireworks 401 → fallback to local model)
- [ ] Stderr shows Fireworks detected the env var

### 3.3 Mode C — Full pipeline coverage (8+ diverse categories)

Tests the official "Return results for every required task" and "Every task ID has an answer" checks.

Use the public validation examples from the official FAQ:

```bash
cat > /tmp/test_c.json << 'ENDJSON'
[
  {"task_id":"T01","prompt":"Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB."},
  {"task_id":"T02","prompt":"A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?"},
  {"task_id":"T03","prompt":"Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived two days late and the packaging was damaged, but the item worked perfectly and customer support resolved my complaint within an hour.'"},
  {"task_id":"T04","prompt":"Summarize the following passage in exactly two sentences: Machine learning is increasingly deployed in healthcare for diagnosis, treatment planning, and patient monitoring. These systems analyse medical images, predict patient deterioration, and spot patterns in electronic health records that might be missed by human clinicians. However, concerns remain about model interpretability, data privacy, liability when errors occur, and the potential for algorithmic bias to worsen existing healthcare disparities. Regulatory frameworks are still catching up with the pace of deployment, creating uncertainty for healthcare providers and technology developers alike."},
  {"task_id":"T05","prompt":"Extract all named entities from the following text and label each as PERSON, ORGANIZATION, LOCATION, or DATE: On March 15 2023, Sundar Pichai announced that Google would open a new AI research lab in Zurich, partnering with ETH Zurich to focus on large language model safety."},
  {"task_id":"t6","prompt":"Write a Python function called is_palindrome that takes a string and returns True if it's a palindrome, False otherwise."},
  {"task_id":"t7","prompt":"Fix the bug in this function: def find_max(lst): max_val = 0; for x in lst: if x > max_val: max_val = x; return max_val. The function returns 0 for lists with all negative numbers."},
  {"task_id":"t8","prompt":"If all A are B, and some B are C, can we conclude that some A are C?"}
]
ENDJSON
mkdir -p /tmp/test_c_out

docker run --rm \
  --cpus=2 --memory=4g --memory-swap=4g \
  -v /tmp/test_c.json:/input/tasks.json:ro \
  -v /tmp/test_c_out:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix 2>&1
echo "Exit code: $?"
```

**Pass criteria:**
- [ ] Exit code 0
- [ ] 8 results in `/tmp/test_c_out/results.json` (1 per task)
- [ ] Every answer is non-empty
- [ ] No task took > 30s (check stderr timestamps)
- [ ] Total runtime < 240s (for 8 tasks)
- [ ] JSON output is valid (`python3 -c "import json; json.load(open('/tmp/test_c_out/results.json'))"`)
- [ ] Required fields present: `task_id`, `answer`

### 3.4 Mode D — Real Fireworks API key

Tests the official Fireworks escalation path (only if you have a key).

```bash
cat > /tmp/test_d.json << 'ENDJSON'
[
  {"task_id":"t1","prompt":"What is the capital of France?"},
  {"task_id":"t2","prompt":"Explain quantum entanglement in one sentence."}
]
ENDJSON
mkdir -p /tmp/test_d_out

# Pass the key via --env-file to avoid terminal masking
echo "FIREWORKS_API_KEY=YOUR_KEY_HERE" > /tmp/fw_key.env

docker run --rm \
  --cpus=2 --memory=4g --memory-swap=4g \
  --env-file /tmp/fw_key.env \
  -e ALLOWED_MODELS="accounts/fireworks/models/deepseek-v4-flash" \
  -e DEADLINE_S=600 \
  -v /tmp/test_d.json:/input/tasks.json:ro \
  -v /tmp/test_d_out:/output \
  ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix 2>&1 | grep -E "Fireworks|answer|error|HTTP"
echo "Exit code: $?"
```

**Pass criteria:**
- [ ] Exit code 0
- [ ] Stderr shows "Fireworks solver ready" (not "No FIREWORKS_API_KEY")
- [ ] No HTTP errors
- [ ] Answers are correct

---

## Phase 4 — Pre-Push Verification

### 4.1 Verify public pull (simulates the grader)
```bash
docker logout ghcr.io
docker pull ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix
```
**Must succeed without auth** — the grader pulls anonymously.

### 4.2 Verify against every FAQ error state

| FAQ Error | Our protection | How to verify |
|-----------|---------------|--------------|
| PULL_ERROR | Image public, tag exact | `docker logout ghcr.io && docker pull <image>` ✅ |
| RUNTIME_ERROR | All imports resolve, no broken symlinks | Mode A passes ✅ |
| TIMEOUT | 1.5B model, 19 tasks ~570s < 600s | Mode C timing check ✅ |
| OUTPUT_MISSING | `_write_output()` called every 5 tasks + at end | Mode A/B/C produce results.json ✅ |
| INVALID_RESULTS_SCHEMA | JSON validated by test | `python3 -c "json.load(open(...))"` ✅ |
| MISSING_TASKS | N results in = N results out | Mode C: 8 in, 8 out ✅ |
| ACCURACY_GATE_FAILED | Answer quality (separate concern) | Mode C: check answers against expected ✅ |
| INFRA_ERROR | Can't control — re-save if happens | — |

### 4.3 Check image size
```bash
docker images ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix \
  --format "{{.Size}}"
# Expected: ~2-3GB (python:3.12-slim + 1.1GB model + deps)
```

---

## Phase 5 — Push & Submit

### 5.1 Push to GHCR
```bash
cat ~/.ghcr-token | docker login ghcr.io -u artemkorolev1 --password-stdin

docker push ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix
docker tag ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix \
  ghcr.io/artemkorolev1/amd-hackathon-submit:latest
docker push ghcr.io/artemkorolev1/amd-hackathon-submit:latest
```

### 5.2 Verify push
```bash
# Check the exact tag exists
docker manifest inspect ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix | head -5
```

### 5.3 Ensure package is PUBLIC
- Go to https://github.com/users/artemkorolev1/packages/container/amd-hackathon-submit/settings
- Under "Danger Zone" → "Change visibility" → Make PUBLIC
- Verify: `docker logout ghcr.io && docker pull ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix`

### 5.4 Submit on lablab.ai
1. Go to the lablab.ai submission form
2. Set Docker image reference to: `ghcr.io/artemkorolev1/amd-hackathon-submit:v14-fix`
3. Click "Update Submission"

**CRITICAL ORDER:** Push → Verify tag exists → Re-save form. Re-saving before the tag exists = PULL_ERROR.

### 5.5 Record the submission
Add to CONTEXT.md:
```
| `v14-fix` | <date+time> CDT | Built ~<time>, pushed ~<time> | Re-saved <tag> | Pending | Import+input fixes only |
```

---

## FAQ Error Recovery

| Error on grader | Action |
|----------------|--------|
| PULL_ERROR | Check GHCR visibility (PUBLIC), check tag name, re-save form AFTER confirming pullable |
| RUNTIME_ERROR | Run Mode A locally. Most likely: import bug missed, or a file that exists locally doesn't in the image |
| INFRA_ERROR | Re-save same form (no rebuild needed) — transient infrastructure glitch |
| TIMEOUT | Run Mode C with timing. If total > 500s for 8 tasks, 19 will exceed 600s |
| ACCURACY_GATE_FAILED | Container works fine — accuracy issue, not build issue. Tune prompts/routing separately |

---

## Files modified by this plan

| File | Change | Risk level |
|------|--------|-----------|
| `harness.py` line 71 | `agent.stage0` → `agent.pre_filter` | Safe — same function name and signature |
| `harness.py` line 72 | `agent.stage2` → `agent.category_filter` | Safe — same function name and return type |
| `harness.py` lines 601-606 | Input path logic: check `/input/tasks.json` first | Safe — reading from correct mount, fallbacks preserved |
| `.dockerignore` (new file) | Excludes stale files from build context | Safe — only reduces what COPY sees |

**Total logic changes: ZERO.** No pipeline routing, no prompts, no models, no config values changed.
