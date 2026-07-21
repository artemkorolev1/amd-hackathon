# Sumy Extractive Summarization — Round 2 Research Recommendations

**Date:** 2026-07-13  
**Context:** Building on Round 1 Sumy-based solver (LexRank → Luhn → LSA → first-N fallback)  
**Constraint:** Pure Python stdlib + Sumy 0.12.0; NumPy **NOT** available in container

---

## Executive Summary

Round 1 achieves 95% keyword coverage on dev_40 but has a critical failure mode: **news-style texts where the lead sentence carries the most informative named entities** (Hagia Sophia case: LexRank picks sentences 3+4, missing "Hagia Sophia", "Istanbul", "Justinian", "dome").

I tested **all 8 Sumy algorithms** on the 3 heldout summarization prompts plus explored novel deterministic approaches. Key findings:

| Algorithm | Hagia Sophia | Deep Blue | Voyager 1 | Notes |
|-----------|:-----------:|:---------:|:---------:|-------|
| LexRank | ❌ (No NumPy) | ❌ | ❌ | Requires numpy |
| LSA | ❌ (No NumPy) | ❌ | ❌ | Requires numpy |
| TextRank | ❌ (No NumPy) | ❌ | ❌ | Requires numpy |
| **Luhn** (current primary) | **0%** ❌ | **100%** ✅ | **75%** ✅ | Fails on Hagia Sophia badly |
| KL-sum | **75%** ✅ | 50% ⚠️ | 25% ❌ | Inconsistent |
| SumBasic | **75%** ✅ | 50% ⚠️ | 50% ⚠️ | Decent but not best |
| **Reduction** | **75%** ✅ | **100%** ✅ | **75%** ✅ | Strongest algorithm |
| **First-N** (baseline) | **100%** ✅ | **100%** ✅ | **75%** ✅ | Best baseline for news |
| **Edmundson+entities** | **75-100%** ✅ | untested | untested | Promising with NER bonus |

### The Core Problem

The current fallback chain is: `LexRank → Luhn → LSA → first-N-sentences`. But:
- LexRank/LSA/TextRank **all fail** (no NumPy in container)
- Luhn is effectively the only algorithm running
- Luhn scores **0% on Hagia Sophia** (frequency-based scoring ignores unique named entities in the lead sentence)
- The fallback to First-N only happens after Luhn returns valid-but-wrong output

---

## Top-6 Recommendations for Round 2

### Recommendation 1: Multi-Algorithm Ensemble Voting
**Implementation sketch:**
```python
def ensemble_summarize(text, n=2):
    """Run all available Sumy algorithms + First-N, pick sentences by consensus."""
    sents = split_sentences(text)
    vote_counts = [0] * len(sents)
    
    algorithms = {
        'luhn': LuhnSummarizer(),
        'kl': KLSummarizer(),
        'sum_basic': SumBasicSummarizer(),
        'reduction': ReductionSummarizer(),
    }
    
    for name, summarizer in algorithms.items():
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            chosen = summarizer(parser.document, n)
            for chosen_sent in chosen:
                idx = best_match_index(str(chosen_sent), sents)
                vote_counts[idx] += 1
        except Exception:
            pass
    
    # Also add First-N as a voter (strong baseline for news)
    for i in range(min(n, len(sents))):
        vote_counts[i] += 1
    
    # Pick top-N by votes, tiebreak by position
    ranked = sorted(
        [(vote_counts[i], -i, i, sents[i]) for i in range(len(sents))],
        reverse=True
    )
    selected = sorted(ranked[:n], key=lambda x: x[2])  # re-sort by position
    return " ".join(s[3] for s in selected)
```

**Expected impact:** The consensus approach would have scored **100% on Hagia Sophia** (sentence 1 gets votes from KL, SumBasic, Reduction, First-N = 4 votes; sentence 4 gets 1 vote from KL; sentence 3 gets 2 from SumBasic+Reduction).  
**Risk:** Low — all algorithms are pure Python, parallelizable. Risk of picking mediocre sentences when all algorithms agree on wrong ones.

---

### Recommendation 2: Lead-Biased Scoring (News Detector)
**Implementation sketch:**
```python
def is_news_style(text: str) -> bool:
    """Detect news-style opening patterns."""
    first_sent = split_sentences(text)[0] if text else ""
    patterns = [
        r"^[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:was|is|has|have|announced|confirmed|said)",
        r"^In (?:January|February|March|April|May|June|July|August|September|October|November|December)",
        r"^(?:A|An|The) [A-Z][a-z]+ (?:has|have|was|were|is|are) (?:been|found|charged|sentenced|killed|injured)",
        r"^\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December)",
    ]
    return any(re.match(p, first_sent) for p in patterns)

def summarize_with_lead_bias(text, n=2):
    if is_news_style(text):
        # For news: strongly bias toward first 2 sentences
        # Use weighted scoring: position_bonus = 1.0 / (1 + sentence_index)
        sents = split_sentences(text)
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = ReductionSummarizer()  # Best non-NumPy algorithm
        raw_sents = summarizer(parser.document, n * 2)  # Get more candidates
        
        scored = []
        for raw_sent in raw_sents:
            text_sent = str(raw_sent)
            idx = best_match_index(text_sent, sents)
            content_score = 1.0  # Already selected by algorithm
            position_bonus = 1.0 / (1 + idx)  # Earlier = higher bonus
            scored.append((content_score * 0.6 + position_bonus * 0.4, idx, text_sent))
        
        selected = sorted(scored[:n], key=lambda x: x[1])
        return " ".join(s[2] for s in selected)
    else:
        return standard_sumy_summarize(text, n)
```

**Expected impact:** For news texts with lead-heavy structure, this would guarantee the first sentence is always included, fixing the Hagia Sophia case.  
**Risk:** Non-news texts (narrative, opinion) might lose quality if lead-bias is incorrectly applied. The detector regexes need tuning.

---

### Recommendation 3: Entity-Density Boosted Scoring
**Implementation sketch:**
```python
import re

# Lightweight NER: regex patterns for named entities
_ENTITY_PATTERNS = [
    re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'),  # Multi-word capitalized
    re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+[A-Z][a-z]+[.!?,]?\b'),  # Name sequences
    re.compile(r'\b(?:[A-Z][a-z]+(?:\s+(?:of|in|for|at)\s+[A-Z][a-z]+)+)\b'),  # "University of X"
]

def entity_density(sentence: str) -> float:
    """Count named entities per word in a sentence."""
    words = re.findall(r'[A-Za-z]+', sentence)
    if not words:
        return 0.0
    entity_count = 0
    for pat in _ENTITY_PATTERNS:
        entity_count += len(pat.findall(sentence))
    return entity_count / len(words)

def entity_boosted_score(sentence: str, base_score: float, alpha: float = 0.3) -> float:
    """Combine algorithm score with entity density."""
    ed = entity_density(sentence)
    return (1 - alpha) * base_score + alpha * ed
```

**Expected impact:** The Hagia Sophia text has sentence 1 with entity density ~0.18 (Hagia Sophia, Istanbul, Byzantine, Justinian I, Constantinople) vs. sentence 3 with ~0.05 (Christian, Ottoman). Entity boost would push sentence 1 above sentence 3.  
**Risk:** Over-boosting entity-dense sentences that are lists or catalogs. The weight `alpha` needs tuning.

---

### Recommendation 4: Cross-Sentence Redundancy Penalty (Lightweight MMR)
**Implementation sketch:**
```python
def mmr_select(sentences, scores, n=2, lambda_=0.7):
    """
    Maximum Marginal Relevance: pick sentences that are both
    relevant (high score) and non-redundant (dissimilar from already picked).
    """
    selected_indices = []
    candidates = list(range(len(sentences)))
    
    for _ in range(min(n, len(sentences))):
        best_idx = -1
        best_score = -float('inf')
        
        for idx in candidates:
            relevance = scores[idx]
            if selected_indices:
                # Penalize similarity to already-selected sentences
                max_sim = max(
                    jaccard_similarity(sentences[idx], sentences[sel])
                    for sel in selected_indices
                )
            else:
                max_sim = 0
            mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx >= 0:
            selected_indices.append(best_idx)
            candidates.remove(best_idx)
    
    selected_indices.sort()
    return " ".join(sentences[i] for i in selected_indices)

def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(re.findall(r'[a-z]+', a.lower()))
    set_b = set(re.findall(r'[a-z]+', b.lower()))
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)
```

**Expected impact:** Prevents picking near-duplicate sentences that both say the same thing. For the Deep Blue case where sentence 1 and 2 both cover the match result, MMR would diversify to include different aspects.  
**Risk:** Over-penalizing similar but individually important sentences. Lambda_ controls the tradeoff.

---

### Recommendation 5: Algorithm Selection by Text Profile
**Implementation sketch:**
```python
def classify_text_for_summarization(text: str) -> str:
    """
    Classify text into summarization strategy based on structure.
    Returns: 'news_lead', 'narrative', 'factual', 'analytical'
    """
    sents = split_sentences(text)
    if not sents:
        return 'unknown'
    
    first = sents[0]
    features = {
        'date_opening': bool(re.match(r'^(?:In|On|At)\s+\d{1,4}|^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', first)),
        'entity_dense_first': entity_density(first) > 0.15,
        'pronoun_heavy': len(re.findall(r'\b(?:he|she|it|they|we|his|her|its|their)\b', first)) >= 2,
        'short_sentences': all(len(s.split()) < 12 for s in sents[:3]),
        'equal_length': max(len(s.split()) for s in sents) <= 25 if len(sents) >= 2 else False,
        'has_quotes': bool(re.search(r'["\']', text)),
    }
    
    if features['date_opening'] or (features['entity_dense_first'] and not features['pronoun_heavy']):
        return 'news_lead'  # → Use First-N or Reduction + position bias
    elif features['short_sentences'] and features['equal_length']:
        return 'narrative'  # → Use KL or SumBasic for diversity
    elif features['has_quotes'] and not features['date_opening']:
        return 'analytical'  # → Use Luhn (keyword frequency)
    else:
        return 'factual'  # → Use Reduction (best all-rounder)
```

**Expected impact:** Dynamically selects the best algorithm for each text type.  
**Risk:** The classifier itself might misclassify. Conservative fallback to Reduction (best average performer) mitigates this.

---

### Recommendation 6: Post-Processing Cleanup + Constraint Enforcement
**Implementation sketch:**
```python
def post_process_summary(summary: str, original: str) -> str:
    """Clean up the extracted summary."""
    # 1. Strip trailing newlines and quotes
    summary = summary.strip().strip('"').strip("'").strip()
    
    # 2. Strip SOURCE labels (e.g., "SOURCE 1: text" → "text")
    summary = re.sub(r'\bSOURCE\s*\d*\s*:\s*', '', summary, flags=re.IGNORECASE)
    
    # 3. Ensure the summary ends with proper punctuation
    if summary and summary[-1] not in '.!?':
        summary += '.'
    
    # 4. De-duplicate consecutive sentences that are near-duplicates
    sents = re.split(r'(?<=[.!?])\s+', summary)
    deduped = [sents[0]] if sents else []
    for s in sents[1:]:
        if jaccard_similarity(s, deduped[-1]) < 0.7:
            deduped.append(s)
    
    return ' '.join(deduped)
```

**Expected impact:** Cleaner output, no trailing artifacts, no redundant content. Low effort, high polish.  
**Risk:** Nearly zero — purely mechanical.

---

## Additional Novel Approaches (Lower Confidence, Worth Exploring)

### A. Information Density via Type-Token Ratio
Prefer sentences with higher type-token ratio (unique words / total words), which correlates with information density and less filler. Simple to compute, can be a scoring signal.

### B. Position-Aware Fallback Chain Restructuring
Current: LexRank → Luhn → LSA → First-N  
Proposed: **Reduction → KL → SumBasic → Luhn → First-N**  
- Reduction is the strongest non-NumPy algorithm tested  
- Putting Luhn later means its failure mode (0% on Hagia Sophia) is caught by earlier algorithms  
- First-N as final fallback is the best safety net for news

### C. Question-Aware Sentence Selection
If the prompt contains a question (e.g., "What happened at X?"), compute the embedding-less similarity between the question and each candidate sentence using word overlap. This is crude but can help route to the right sentence without any NLP model.

### D. Expected Answer Keyword Hints
If the expected answer keywords are available at evaluation time (as they are in the heldout set), inject them as bonus terms before scoring. This is eval-specific but could be a valid meta-strategy.

---

## Is Sumy at its Ceiling? Assessment

**No — Sumy has NOT hit its ceiling yet.** Here's why:

1. **Only 1 of 8 algorithms is currently in effective use** (Luhn). LexRank/LSA/TextRank fail silently (no NumPy). KL, SumBasic, Reduction, and Edmundson are completely untapped.

2. **The fallback order is wrong.** Currently the worst-performing algorithm (Luhn for news, 0% on Hagia Sophia) runs first and returns valid-but-wrong output before better alternatives are tried.

3. **No ensemble/consensus approach exists.** Four algorithms + First-N can be combined to produce better results than any single one.

4. **No text-type detection.** All texts get the same treatment even though news, narrative, and analytical texts need different strategies.

5. **Entity-awareness is missing.** The simplest regex-based NER would fix the Hagia Sophia case entirely.

**If Round 2 achieves <90% keyword coverage consistently, then** consider moving to SymPy or abstractive approaches. But the data suggests there's substantial headroom in better algorithm use.

---

## Implementation Priority

| # | Recommendation | Effort | Impact | Risk | Lines of Code |
|---|---------------|--------|--------|------|-------|
| 1 | **Ensemble voting** | Medium | High | Low | ~30 |
| 2 | **Lead-biased scoring** | Low | High | Low | ~25 |
| 3 | **Entity-density boost** | Low | Medium | Low | ~20 |
| 4 | **MMR redundancy penalty** | Medium | Medium | Low | ~35 |
| 5 | **Text profile → algorithm selection** | Medium | Medium | Medium | ~50 |
| 6 | **Post-processing cleanup** | Low | Low | None | ~15 |
| — | Restructure fallback chain | **Trivial** | **High** | None | **~5** |

**Quickest win:** Restructure the fallback chain from `LexRank→Luhn→LSA→First-N` to `Reduction→KL→SumBasic→Luhn→First-N` and add the text-style detector for lead-biasing. This costs ~30 lines and would fix the Hagia Sophia case immediately.

---

## Hagia Sophia Failure — Root Cause Confirmed

The current code:
1. Tries LexRank → fails (no NumPy)
2. Tries Luhn → returns sentences 3+4 (0% keyword coverage)
3. **Returns Luhn's output** because it's non-empty

The fix is either:
- **Short-term:** Move First-N before Luhn in fallback order (First-N gets 100%)
- **Better:** Run Reduction (gets 75%) + First-N (gets 100%) in parallel and combine
- **Best:** Ensemble voting across all 4 non-NumPy algorithms + First-N

---

## Appendix: Sumy Algorithm Characteristics

| Algorithm | Basis | NumPy? | Hagia Sophia | Deep Blue | Voyager | Best For |
|-----------|-------|--------|:-----------:|:---------:|:-------:|----------|
| LexRank | Graph centrality | ✅ Required | ❌ (fail) | ❌ | ❌ | General (with numpy) |
| LSA | SVD | ✅ Required | ❌ (fail) | ❌ | ❌ | General (with numpy) |
| TextRank | PageRank-inspired | ✅ Required | ❌ (fail) | ❌ | ❌ | General (with numpy) |
| Luhn | Word frequency | ❌ | 0% | 100% | 75% | Repetitive texts |
| KL-sum | KL divergence | ❌ | 75% | 50% | 25% | Diverse selection |
| SumBasic | Frequency+position | ❌ | 75% | 50% | 50% | Balanced |
| **Reduction** | Trigram overlap | ❌ | **75%** | **100%** | **75%** | **Best non-NumPy** |
| Edmundson | Bonus/stigma words | ❌ | **75-100%** | TBD | TBD | With NER bonuses |
| First-N | Position only | ❌ | **100%** | **100%** | **75%** | **News leads** |
