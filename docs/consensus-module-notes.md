# Self-Consistency Voting Module

## Location
`agent/solvers/local_vote.py` — refactored plug-and-play module.

## Architecture Decision

### Problem
Harness used single `_infer()` call with `temperature=0.0`. No self-consistency, no quality assessment before deciding whether to escalate to Fireworks.

### Solution
Self-consistency voting: sample the model k times with varying temperatures, normalise answers per category, majority vote, escalate on low agreement.

### Key Constraint
Grader hard-kills at 30s per question. On CPU (~12 tok/s), 3 samples × 150 tokens = 37.5s theoretical. Solution:
- Default OFF (`CONSENSUS_SAMPLES=1` in Dockerfile)
- Only apply to fast short-answer categories: `math`, `sentiment`, `ner`
- `budget_per_sample = 30.0 / k` guarantees total stays under limit
- Skip entirely when reasoning model loaded (`_REASONING_HEADROOM > 1`)

### Critical Blocker Avoided
Both `harness.py` and the old `local_vote.py` independently loaded `llama_cpp` with different default GGUF models. That would OOM on 4GB RAM. Fix: pass the already-loaded `llm` instance as a parameter. The module no longer has its own model loader.

## Integration Contract

```python
result = solve_with_consensus(
    llm=llm,                    # pre-loaded Llama (caller owns lifecycle)
    prompt=prompt,
    category=category,
    system_prompt=sys_prompt,
    k=CONSENSUS_SAMPLES,
    max_tokens=max_tok,
    timeout_per_sample=budget_per_sample,
)
```

Returns `{"majority_answer", "agreement_score", "all_answers", "samples"}`.

## QC Gate Chain
1. solve_with_consensus() returns majority_answer (raw text)
2. verify(answer, category) checks for hedge words, degeneracy, emptiness
3. If verify fails OR agreement_score < 0.5 → _fw_fallback() escalates to Fireworks

## Usage
```bash
# Default (no voting, safe for Docker submission)
docker run -e CONSENSUS_SAMPLES=1 ...

# Enable 3-way voting for local dev
CONSENSUS_SAMPLES=3 python3 harness.py eval_mini_10.json
```

## Competitor Analysis
None of the three researched competitors (KaananeTaha, AdityaAlfaaz, luongs3) use self-consistency voting. This is a differentiator.

## Promising Future Adoptions
1. `_FB` fallback suffix: "If the request does not match this description, just answer it directly and accurately" — conditional for sentiment/NER where format mismatch hurts most
2. Per-category `reasoning_effort`: "adaptive" for math/logic/code, "none" for sentiment/NER/factual — currently we set "none" globally
3. Reasoning preamble stripper: remove `**Analyze**`, `**Step**` markers from GLM model outputs
4. Model fallback cascade: prioritized list of FW models per category (like luongs3)
