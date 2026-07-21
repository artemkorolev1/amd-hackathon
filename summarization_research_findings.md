# Sub-1B LLMs for Headline Generation: Research Findings

## Problem Analysis

**Task:** 19 xsum-format questions in training-v3.json. Given ~300 chars of BBC news text with "Summarize:" prefix, output a 60-150 character headline capturing the core event with exact names/numbers.

**Current State:** 0% accuracy across ALL models (Qwen2.5-1.5B, Gemma-3-1B, SmolLM2-1.7B, Llama-3.2-1B, Qwen2.5-Coder-1.5B, Qwen2.5-Math-1.5B) and ALL prompt variants tried.

## What Has Already Been Tested (Exhaustive List)

### Prompt strategies tried:
1. **Basic summarization prompts**: "Summarize in at most 2 sentences. Include key names and numbers."
2. **Bullet-point summaries**: "Summarize as bullet points."
3. **Terse prompts**: "Summarize:" or "One sentence summary:"
4. **Headline framing**: "What is the headline for this news story?"
5. **Title-only**: "Write a title for this text. Just the title."
6. **Key facts extraction**: "Extract: who did what, when, and where?"
7. **Chain-of-thought**: "Find the most important information. Who is involved? Write a one-sentence headline."
8. **Empty system prompt**: No system prompt at all.
9. **Key-points framing**: "Key points from this text:"
10. **Empty prompt + Gemma**: Blank system prompt.

### Models tested:
- Qwen2.5-1.5B (generalist)
- Gemma-3-1B (summarization specialist candidate)
- SmolLM2-1.7B
- Llama-3.2-1B
- Qwen2.5-Coder-1.5B
- Qwen2.5-Math-1.5B

### Evaluations run (all 0% for summarization):
- Round 1: 4 variants on Gemma-1B (basic, bullet, terse, direct)
- Round 2: 4 variants on Qwen-1.5B (2-sentences, facts, bullet, terse)
- Round 3: 8 variants on Qwen + Gemma (incl. empty, key-facts, bullet, one-line, extract)
- Round 4: 4 variants on Coder + Qwen (summary, key-points, who-what)
- CoT: 8 variants on SmolLM2 + Llama (headline, who-what, title-only, cot-headline)
- Final config ablation: Coder with "Key points from this text:" (0%)
- Three-model baseline eval: Qwen2.5-1.5B with generic system prompt (0% summarization, 37.5% overall)
- Ensemble router: Per-category specialist (0% summarization)

## Root Cause Analysis

### Failure Pattern #1: Multi-sentence factual summary instead of headline
All models produce 2-3 sentence summaries of the article content, not single-sentence headlines. Example:
```
Expected: Peterborough United defender Miles Addison has signed a new one-month contract with the League One side.
Got:      Graham Westley's Posh side has had a quiet season, with one goal in a recent match against Barnsley, while facing injuries to key defenders Callum Elder, Gabriel Zakuani, and Kgosi Ntlhe. Posh are playing at home against struggling Chesterfield on Boxing Day.
```

### Failure Pattern #2: "Here is a summary..." preamble
Many model outputs start with meta-commentary like "Here's a summary of the provided text:" or "Here are the key facts:" — wasting tokens and never producing the headline format.

### Failure Pattern #3: Information selection mismatch
The models distribute attention across all details in the prompt. The expected headline requires selecting the SINGLE most newsworthy event/conclusion. The models instead enumerate multiple facts with equal weight.

### Failure Pattern #4: Unable to infer the headline from the article body
The xsum dataset's first sentence IS the headline condensation, but the body provides supporting details. The model reads the body and summarizes it factually, missing the "lede" that a human journalist would write.

### Failure Pattern #5: Fuzzy matching bar is unreachable
The 80% token overlap threshold requires the output to share 80% of the expected answer's tokens. Since the model outputs are completely different text (same topic, different phrasing/selection), overlap is consistently 0-20%.

## Why Previous Strategies Failed

| Strategy | Why It Failed |
|----------|--------------|
| "Summarize in 2 sentences" | Models produce 2+ sentences of factual summary, not 1 headline sentence |
| "Headline" / "Title" | Models still generate multi-sentence summaries even when told "headline" |
| "Extract who/what/where" | Models extract detailed facts but format as a paragraph |
| CoT (step by step) | Models think step-by-step then output verbose summary, not headline |
| Empty system prompt | Models default to their training behavior (multi-sentence summary) |
| Few-shot (in CoT round) | Not enough examples to shift behavior; 2 examples lost in context |
| Prefill (not tested) | Could force format but likely to produce partial/incomplete text |

## Concrete Recommendations

### Recommendation 1 (BEST): Fireworks API Fallback for Summarization

**Verdict: STRONGLY RECOMMENDED**

The kimi-k2p7-code model on Fireworks is far more capable than any 1B model. With only 19 summarization questions, API costs would be negligible.

**Implementation:**
```python
fw_solver = FireworksSolver()  # reads FIREWORKS_API_KEY from env
answer = fw_solver.solve(
    model="accounts/fireworks/models/kimi-k2p7-code",
    system_prompt="You are a news headline writer. Read the article and write ONE headline sentence (max 20 words) that captures the core event. Use exact names, numbers, locations. Output ONLY the headline, no prefix, no explanation.",
    user_prompt=prompt,  # "Summarize: ..."
    max_tokens=80,
    temperature=0.0,
    task_type="summarization",
)
```

**Recommended system prompt for kimi:**
```
You write BBC-style news headlines. Given an article, output a single headline sentence (8-20 words) capturing the main event. Use exact names, numbers, places. Do NOT start with "Headline:" or any prefix. Just the headline text.
```

**Anticipated accuracy: >80%** based on the v12d results that hit 93.7% using stronger models.

### Recommendation 2: Aggressive Format Enforcement via Prefill + Strict Token Budget

**Verdict: WORTH TESTING, moderate confidence**

Force the model into a single-sentence format using prefill and strict length constraints.

**Prompt:**
```
System: Write a one-sentence news headline that captures the MOST important event.
```
```
User: Summarize: <article text>
Assistant: HEADLINE: <-- prefill this
```

Key parameters:
- `prefill="HEADLINE: "` (forces output to start with this)
- `max_tokens=80` (forces brevity — headline must fit in ~15 words)
- `temperature=0.0`
- `stop=["\n"]` (stop at first newline, ensuring single sentence)

**Model to use: Qwen2.5-1.5B** (most instruction-following of the 1B models)
**Anticipated accuracy: ~10-30%** — the model might produce a short answer but may still miss the right event selection.

### Recommendation 3: Two-Step Extractive Pipeline

**Verdict: MODERATE CONFIDENCE, complex but could work**

Step 1: Extract key entities + core event from the article
Step 2: Format into a headline template

**Step 1 prompt (system):**
```
Extract the following from the news text in this exact format:
WHO: [main person, group, or organization involved]
WHAT: [the single most important action or event]
WHERE: [location]
RESULT: [what happened as a result]

Output ONLY these four fields. No explanation.
```

**Step 2 prompt (system):**
```
Given the extracted WHO, WHAT, WHERE, and RESULT, write a news headline in the format: "WHO has WHAT, WHERE, RESULT"
Example:
WHO: Miles Addison
WHAT: signed a new one-month contract
WHERE: League One side Peterborough United
RESULT: (no specific result)
→ Headline: Peterborough United defender Miles Addison has signed a new one-month contract with the League One side.

Now write the headline:
WHO: {extracted_who}
WHAT: {extracted_what}
WHERE: {extracted_where}
RESULT: {extracted_result}
```

**Model: Qwen2.5-1.5B** for both steps
**Anticipated accuracy: ~20-40%** — step 1 is feasible, step 2 formatting is the challenge

### Recommendation 4: Few-Shot with 3-4 Examples in the User Message

**Verdict: WORTH TESTING, moderate confidence**

Since few-shot in the system prompt was ineffective (probably due to context position), put examples in the user message.

**Prompt:**
```
System: You write BBC-style news headlines. Output ONLY the headline, one sentence, max 20 words.

User: Article: Two men were assaulted in a house in south Belfast by a gang armed with a knife, hammer and batons. The gang stole cash and personal items.
Headline: Two men have been assaulted by an armed gang in south Belfast.

Article: A 33-year-old Leicester City striker has only played twice this season. He scored 3 goals in 23 games last season for Blackpool. He joined Sheffield Wednesday this week.
Headline: Leicester City striker Gary Taylor-Fletcher has joined Sheffield Wednesday on an initial month-long loan.

Article: {actual article text}
Headline:
```

**Model: Qwen2.5-1.5B**
**Anticipated accuracy: ~10-20%** — tiny models are weak at following few-shot patterns in-context

### Recommendation 5: Restructure the Task Completely

**Verdict: WORTH TESTING, low-moderate confidence**

Instead of "summarize" or "headline", frame this as a **question answering** task:

**Prompt:**
```
System: Answer the question about the news article.
User: Article: {article text}
Question: What is the single most important thing that happened?
Answer:
```

Or even more direct:
```
System: Read the article. Then answer: "What happened?" in one short sentence.
```

**Model: Qwen2.5-1.5B**
**Anticipated accuracy: ~5-15%** — the "what happened" framing might focus the model on the key event

### Recommendation 6: Pattern-Matched Headline Templates (Heuristic)

**Verdict: LOW CONFIDENCE, last resort**

Analyze the model's output text and extract a headline from it using regex/NLP heuristics rather than expecting the exact format.

**Approach:**
1. Run the model with any summarization prompt
2. Try to extract a headline from the output:
   - Take the first sentence if it's under 200 chars
   - Look for sentences with "has/have" (BBC headline pattern): `re.findall(r'[A-Z][^.!?]*(?:has|have)[^.!?]*[.!?]', output)`
   - Try to match known entity patterns
3. This is fragile but might salvage some correct event+entity combinations

**Anticipated accuracy: ~5%** — heuristics are never reliable enough for 80% token overlap

## Summary of Expected Outcomes

| Approach | Model | Est. Accuracy | Effort | Best For |
|----------|-------|:------------:|:------:|----------|
| Fireworks API fallback | kimi-k2p7-code | **>80%** | Low | Production quality |
| Prefill + strict length | Qwen2.5-1.5B | 10-30% | Low | Quick test |
| Two-step extractive | Qwen2.5-1.5B | 20-40% | Medium | Research |
| Few-shot in user msg | Qwen2.5-1.5B | 10-20% | Low | Quick test |
| "What happened?" QA | Qwen2.5-1.5B | 5-15% | Low | Quick test |
| Heuristic extraction | Any | ~5% | Medium | Last resort |

## Key Insight: Why 1B Models Can't Do Abstractive Headlines

The xsum task requires:
1. **Information prioritization**: Identify the single most newsworthy element from multiple facts
2. **Abstractive phrasing**: Generate novel text that isn't a direct extract
3. **Length discipline**: Exactly 60-150 characters
4. **Entity accuracy**: Exact names, numbers, dates

Sub-1B models lack sufficient capacity for (1) and (2). Their attention mechanism distributes weight evenly across input tokens, so all facts get equal treatment. The "Summarize:" prefix triggers their training distribution of producing factual multi-sentence summaries.

The report showing qwen2.5-1.5b at 75% on summarization for the 300-set likely used a different dataset with either:
- Simpler/extractive summarization tasks
- Different evaluation metric
- Different text formats (not xsum-style headline generation)

**Bottom line**: For abstractive headline generation on xsum-style data, 1B models fundamentally cannot do this task without either (a) fine-tuning, or (b) an API fallback to a much stronger model.
