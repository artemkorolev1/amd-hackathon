# Secondary Summarization Classifier — Design Document

## Problem Statement

The primary 8-way scorer (`agent/category_filter.py`) misroutes summarization prompts
to **math**, **code_gen**, and **logic** when explicit "summarize" keywords are absent.
The existing harness overrides (doc headers, explicit keywords, long-text guard) catch
some but not all cases — summarization accuracy in the per-category GPU eval is only
**37.8%** (14/37), the worst of all 8 categories.

## Root Causes

### 1. Summarization → Math
**Trigger**: Range expressions in news prose ("10-20 shots per second", "2.05 megajoules -
3.15 megajoules", "$700 million in pledges") trigger `num_op` in `_score_math()`. The
hyphen in "10-20" is classified as a subtraction operator despite the `raw_ops` guard
(which only exempts patent/code ranges and age ranges). The long-text guard in math
(>80 words → suppress) is **bypassed when `num_op` is true** (line 178: `if len(words) > 80
and not (explicit or word_prob)` — `num_op` is not checked).

**Fix location**: Math's long-text guard should also suppress when only `num_op` fires
without `explicit` or `word_prob`.

### 2. Summarization → Code Gen
**Trigger**: Technical/prose articles containing words like "return", "function", "class",
"import" in narrative context (e.g., "returns True if", "the function of the liver",
"class of drugs called"). `_score_code_gen()` line 758 fires on `return\b` even in
prose when not in SQuAD format.

**Fix location**: Code gen needs a "prose context" guard — if the word "return" appears
before "to" (as in "returns True if the model is...") or after a 3rd-person subject,
it's not code.

### 3. Summarization → Logic
**Trigger**: Analysis/reasoning prose ("examines", "investigates", "analyzes") combined
with conditional language ("if...then") in document text triggers `_score_logic()`.
Diplomatic statements with "if X then Y" reasoning are particularly vulnerable.

### 4. Summarization → Factual
**Trigger**: News articles starting with question-like sentences or containing "define",
"explain", "what is" in article context.

## Design: Secondary Summarization Classifier

### Architecture

```
classify(prompt) → (category, confidence, scores)
  ↓
if category != "summarization":
    secondary_summarization(prompt, scores) → corrected_category
    if corrected_category == "summarization":
        override primary
```

The secondary classifier is a **pure deterministic module** (`agent/secondary_summarization.py`)
that takes the prompt + the raw scores from the 8-way scorer and returns a corrected
category. It is **conservative** — it only overrides when summarization signals are
unambiguous. It follows the same pattern as `agent/secondary_factual.py`.

### Signal Families

Below are 8 families of signals, each with concrete regex patterns.

---

## Signal Family 1: Multi-Source Document Structure

**Signal strength**: HIGH (3.0–4.0)
**Catches**: Prompts with 2+ numbered sources/studies that lack explicit "summarize" keyword

```
Pattern: Two or more SOURCE / STUDY / DOCUMENT N markers
  r"\b(?:SOURCE|STUDY|DOCUMENT|ARTICLE|REPORT)\s+\d+"   → count >= 2

Pattern: Source attribution with parenthetical
  r"(?:SOURCE|STUDY)\s+\d+\s*\([A-Z][A-Za-z\s]+,\s*\d{4}\)"   → "SOURCE 1 (UNFCCC Press, 2023):"

Pattern: Mixed-case "Source N" (not just uppercase)
  r"^(?:Source|SOURCE)\s+\d+"   → "Source 1" or "SOURCE 1" at line start

Pattern: "SOURCE N (Org, Year):" followed by quoted text
  r"(?:SOURCE|STUDY)\s+\d+\s*\([^)]+\)\s*:\s*['\u201C]"    → Source attribution with opening quote

Suppression guard: If text has only 1 source AND is a factual QA format, don't boost.
```

## Signal Family 2: Document / Legal / Formal Headers

**Signal strength**: HIGH (3.0)
**Catches**: Documents with formal headers misclassified as logic/factual

```
Pattern: All-caps document type header at line start
  r"^(?:LEGAL BRIEF|STATEMENT BY|PRESS RELEASE|EXECUTIVE SUMMARY|"
  r"WHITE PAPER|POLICY BRIEF|MEMORANDUM|TRANSCRIPT|MEMO|BRIEF|"
  r"DECLARATION|AFFIDAVIT|PETITION|MOTION|OPINION|RULING)\b"

Pattern: All-caps multi-word headline (>10 chars) followed by newline
  r"(?:^|\n)\s*[A-Z][A-Z\s'\"-]{10,}\s*:?\s*\n"

Pattern: Diplomatic/formal opening
  r"\b(?:My delegation|The Permanent Representative|"
  r"We reaffirm|We take note|We call upon|My government maintains)\b"
```
Note: Many of these are already partially checked in `_score_summarization()` but
with lower weight. The secondary classifier should use **higher weights** and **no
gate** (no `_has_summary_context` requirement).

## Signal Family 3: News Dateline / Article Opening

**Signal strength**: HIGH (2.5–3.0)
**Catches**: News articles starting with a date (no explicit summarization keyword)

```
Pattern: "On [Month] [Day], [Year]" at start of prompt
  r"^On\s+(?:January|February|March|April|May|June|July|August|"
  r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"

Pattern: "HEADLINE:" followed by quoted sensational text + article body
  r"HEADLINE:\s*['\u201C][A-Za-z\s'!?-]+['\u201D]"

Pattern: News lead — specific person/place + event + time
  r"\b(?:A|An|The)\s+\d+-year-old\b"  → "A 19-year-old man"
  r"\b[A-Z][a-z]+\s+(?:has\s+been|has\s+appeared|has\s+signed|"
  r"is\s+facing|was\s+rescued|was\s+injured|was\s+killed|"
  r"has\s+been\s+charged|has\s+been\s+sentenced)\b"
```

**Example captured**:
- `q-01aac354`: "Apple framed its USB-C adoption as a voluntary innovation choice..."
- `xsum-*`: BBC-style news articles

## Signal Family 4: Synthesize / Multi-Source Instruction Patterns

**Signal strength**: HIGH (3.0–4.0)
**Catches**: The most common summarization instruction variant that lacks the word "summarize"

```
Pattern: "Synthesize" as instruction verb
  r"\bSynthesize\s+(?:the|these|all)\s+(?:three|four|two|five|six|\d+)\s+(?:sources?|articles?|texts?|documents?|studies?|passages?)"

Pattern: "Capture the core" / "balanced summary"
  r"\b(balanced\s+summary|capture\s+the\s+core|capture\s+the\s+main|"
  r"core\s+(?:contradictions|arguments|findings|points|themes)|"
  r"key\s+(?:takeaways|findings|insights|arguments|differences)|"
  r"main\s+(?:arguments|points|findings|differences)|"
  r"critical\s+(?:analysis|assessment|evaluation|review))\b"

Pattern: "Provide a [concise/brief/detailed] [summary/overview/synthesis]"
  r"\b(?:Provide|Give|Write|Offer|Present)\s+(?:a|an)\s+"
  r"(?:concise|brief|detailed|comprehensive|balanced|critical|objective)?\s*"
  r"(?:summary|overview|synthesis|analysis|review|recap)\b"
```

## Signal Family 5: Quoted Speech / Direct Attribution Blocks

**Signal strength**: MEDIUM–HIGH (2.0–3.0)
**Catches**: Documents with substantial quoted/attributed content

```
Pattern: Paragraph-length quoted content (multiple lines of quotes)
  Paragraph count inside quotes: count of '...' or "..." or \u201C...\u201D spans
  r"['\u2018\u2019][^'\u2018\u2019]{50,}['\u2018\u2019]"   → 2+ occurrences
  r"['\u201C][^'\u201C]{60,}['\u201D]"                     → 2+ occurrences

Pattern: Speech attribution verbs near quotes
  r"\b(?:said|stated|declared|argued|claimed|asserted|"
  r"announced|warned|cautions?|emphasized|noted|observed|"
  r"highlighted|explained|added|continued|replied|responded)\b"

Pattern: Attribution + colon + opening quote
  r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*:\s*['\u201C]"

Pattern: Multi-line quoted block (indented or set apart)
  r"(?:^|\n)\s*['\u201C].{100,}['\u201D]"                → multi-line quote block
```

**Example**: The diplomatic UN statement prompt — 3 paragraphs entirely inside quotes.

## Signal Family 6: Citation / Academic Reference Patterns

**Signal strength**: MEDIUM (1.5–2.5)
**Catches**: Academic paper summaries, patent analysis, technical report summaries

```
Pattern: Parenthetical (Author, Year) citations
  r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.|and\s+[A-Z][a-z]+))?,\s*\d{4}[a-z]?\)"

Pattern: "published in" + journal
  r"\bpublished\s+in\s+(?:the\s+)?(?:journal|proceedings|"
  r"Nature|Science|Cell|The\s+Lancet|NEJM|JAMA|"
  r"Physical\s+Review|IEEE|ACM|Elsevier|Springer)\b"

Pattern: "titled" + quoted title
  r"\btitled\s+['\u201C][A-Z][A-Za-z\s:,-]+['\u201D]"

Pattern: Volume/issue/page references
  r"\b(?:Vol\.\s*\d+|Volume\s+\d+|no\.\s*\d+|pages?\s+\d+[-]\d+|"
  r"\d+:\d+[-]\d+)\b"    → "42:315-328"
```

**Example**: Patent filings, biomedical abstracts with citation references.

## Signal Family 7: Statistical / Technical Prose (Numbers in Context)

**Signal strength**: MEDIUM–HIGH (2.0)
**Catches**: Long articles with dense numbers used as statistics, not calculations

This is the **key pattern for math → summarization correction**. It measures the
ratio of "numbers used as facts" to "numbers used in computation."

```
Pattern: Percentage expressions (are statistics, not math)
  r"\d+(?:\.\d+)?%\s+(?:of|of\s+the|reduction|increase|drop|rise|"
  r"of\s+patients|of\s+participants|of\s+the\s+population)"
  → "38% reduction", "17.3% of patients", "27% of participants"

Pattern: Units of measurement (scientific context)
  r"\d+(?:\.\d+)?\s*(?:megajoules|gigawatts|kilowatts|terawatts|"
  r"nanometers|micrometers|millimeters|centimeters|kilograms|"
  r"grams|milligrams|micrograms|liters|milliliters|gallons|"
  r"degrees?\s*(?:Celsius|Fahrenheit|Kelvin)|"
  r"dollars?|euros?|pounds?|yen|cubic\s+meters?|"
  r"per\s+second|per\s+minute|per\s+hour|per\s+day|per\s+year)\b"
  → "2.05 megajoules", "10-20 per second", "$26,500"

Pattern: Monetary ranges and estimates
  r"\$\s*\d+(?:[,.]\d+)?\s*(?:million|billion|trillion)"
  r"\d+(?:\.\d+)?\s*-[–]\s*\d+(?:\.\d+)?\s*(?:percent|%)"
  → "10-20%" as range, not subtraction

Pattern: Ratio of NUMBERS to MATH_VERBS > threshold
  Count: all digits /\d+/
  Count: math verbs (calculate, compute, solve, equation, formula, etc.)
  If nums >= 8 AND nums/math_verbs > 5 AND length > 150 words
  → Strong summarization signal, suppress math
```

### Concrete regex for range-in-prose detection (math guard)

The critical fix for math suppression:

```python
# In the secondary classifier (or as patch to _score_math):
# Range expressions like "10-20 per second", "$700 million - $1 billion"
_ranges_as_rates = re.search(
    r"\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*"
    r"(?:per\s|percent|%|of|megajoules|kilowatts|dollars|"
    r"euros|pounds|people|patients|participants|years|months|days|"
    r"shots|times|sessions|episodes|incidents|cases|deaths|"
    r"injuries|units|items|products|samples|tests|trials)",
    lower
)
if _ranges_as_rates and len(words) > 80:
    # This is a prose range (statistic), not math subtraction
    suppress_math = True
```

## Signal Family 8: Suppression Patterns (Anti-Signals)

These patterns tell us the prompt is NOT summarization even if it has some document-like
structure. Used as negative guards.

```
Pattern: Clear code fence (```python```) → not summarization
  r"```\s*(?:python|java|javascript|rust|go|ruby|typescript|bash)\s*\n"

Pattern: Clear math problem with sequential numbered sub-questions
  r"(?:a\)|b\)|c\)|d\)|e\)|i\)|ii\)|iii\))\s*(?:Calculate|Find|Solve|Compute)"

Pattern: MCQ choices (A. ... B. ... C. ... D. ...)
  r"^(?:A[.)]|B[.)]|C[.)]|D[.)])\s+.+$" in multiline mode with 3+ matches

Pattern: Knight/Knave puzzle — unambiguous logic
  r"\b(knights?|knaves?|always\s+tell(|s)\s+the\s+truth|always\s+lies?)\b"
```

## Recommended Implementation Plan

### A. New module: `agent/secondary_summarization.py`

Follow the pattern of `secondary_factual.py`: a single `resolve_summarization()`
function that takes `(primary_category: str, prompt: str, scores: dict) -> str`.

```python
def resolve_summarization(
    primary_category: str,
    prompt: str,
    scores: Optional[Dict[str, float]] = None,
) -> str:
    """
    Secondary summarization detector — catches prompts that the primary 8-way
    scorer misroutes from summarization to math/code_gen/logic/factual.

    Args:
        primary_category: The category from primary 8-way scorer
        prompt: The full user prompt
        scores: Raw scores from primary scorer (optional)

    Returns:
        Corrected category (returns primary_category if no override)
    """
```

### B. Signal scoring logic

Use weighted scoring. If `_total_summarization_score` > threshold (e.g., 3.0)
AND `_total_summarization_score` >= primary winner's score, override to summarization.

### C. Integration

Add to `agent/classifier.py` and `agent/hierarchical_classifier.py` as a new
secondary classifier step, analogous to `secondary_factual`:

```python
# In hierarchical_classifier.py classify():
if primary_cat != "summarization":
    from agent.secondary_summarization import resolve_summarization
    corrected = resolve_summarization(primary_cat, prompt, result.get("raw_scores"))
    if corrected != primary_cat:
        primary_cat = corrected
        method = "summarization_secondary"
```

### D. Suggested weights for signal families

| Family | Signal | Weight | Gate |
|--------|--------|--------|------|
| 1 | Multi-source (2+ SOURCE/STUDY) | +3.0 | None |
| 1 | Source attribution with paren | +2.0 | None |
| 2 | Legal/doc header | +3.0 | None |
| 3 | News dateline start | +3.0 | None |
| 3 | BBC-style news lead | +2.5 | Not in factual QA format |
| 4 | "Synthesize" + N sources | +4.0 | None |
| 4 | "Capture the core" | +3.0 | None |
| 4 | "Provide a ... summary" | +3.0 | No code signals |
| 5 | Multi-line quoted block | +2.5 | 2+ paragraph quotes |
| 5 | Attribution + quote | +1.5 | Combined with other |
| 6 | Citation (Author, Year) | +2.0 | No clear code |
| 6 | "published in" + journal | +2.0 | No clear code |
| 7 | Statistical numbers (ratio) | +2.0 | word_count > 120, math_score < 3 |
| 7 | Range-as-statistic | +2.0 (to summarization), -2.0 (from math) | word_count > 60 |
| 8 | Code fence present | -5.0 (negate) | Absolute blocker |
| 8 | Knight/knave puzzle | -5.0 (negate) | Absolute blocker |

### E. Concrete regex patterns for the module

```python
import re
from typing import Dict, Optional, Tuple

# ── Family 1: Multi-Source ──
_RE_MULTI_SOURCE = re.compile(
    r"^(?:SOURCE|STUDY|DOCUMENT|ARTICLE|REPORT)\s+\d+\b",
    re.MULTILINE,
)
_RE_SOURCE_ATTRIB = re.compile(
    r"(?:SOURCE|STUDY)\s+\d+\s*\([A-Z][A-Za-z0-9\s&,.]+,\s*\d{4}\)\s*:"
)
_RE_MIXED_SOURCE = re.compile(
    r"^(?:Source|SOURCE)\s+\d+", re.MULTILINE
)
_RE_QUOTED_SOURCE = re.compile(
    r"(?:SOURCE|STUDY)\s+\d+\s*\([^)]+\)\s*:\s*['\u201C]"
)

# ── Family 2: Document Headers ──
_RE_DOC_HEADER = re.compile(
    r"^(?:LEGAL BRIEF|STATEMENT BY|PRESS RELEASE|EXECUTIVE SUMMARY|"
    r"WHITE PAPER|POLICY BRIEF|MEMORANDUM|TRANSCRIPT|MEMO|BRIEF|"
    r"DECLARATION|AFFIDAVIT|PETITION|MOTION|OPINION|RULING)\b",
    re.MULTILINE,
)
_RE_DIPLOMATIC = re.compile(
    r"\b(?:My delegation|The Permanent Representative|"
    r"We reaffirm|We take note|We call upon|My government maintains)\b"
)
_RE_ALLCAPS_HEADER = re.compile(
    r"(?:^|\n)\s*[A-Z][A-Z\s'\"-]{10,}\s*:?\s*(?:\n)"
)

# ── Family 3: News Dateline ──
_RE_DATELINE_START = re.compile(
    r"^On\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
)
_RE_HEADLINE_FORMAT = re.compile(
    r"HEADLINE:\s*['\u201C][A-Za-z\s'!?,-]+['\u201D]"
)
_RE_NEWS_LEAD = re.compile(
    r"\b(?:A|An|The)\s+\d+-year-old\b|\b"
    r"(?:has\s+been\s+(?:charged|arrested|sentenced|found|accused|"
    r"rescued|injured|killed)|"
    r"is\s+facing\s+(?:charges|trial|accusations)|"
    r"has\s+(?:signed|joined|agreed|pledged|announced|confirmed))\b"
)

# ── Family 4: Synthesize / Instruction ──
_RE_SYNTHESIZE = re.compile(
    r"\bSynthesize\s+(?:the|these|all)\s+(?:three|four|two|five|six|\d+)\s+"
    r"(?:sources?|articles?|texts?|documents?|studies?|passages?)"
)
_RE_CORE_CAPTURE = re.compile(
    r"\b(?:balanced\s+summary|capture\s+the\s+(?:core|main|key)|"
    r"key\s+(?:takeaways|findings|insights|arguments|differences)|"
    r"main\s+(?:arguments|points|findings|differences)|"
    r"critical\s+(?:analysis|assessment|evaluation|review))\b"
)
_RE_PROVIDE_SUMMARY = re.compile(
    r"\b(?:Provide|Give|Write|Offer|Present)\s+(?:a|an)\s+"
    r"(?:concise|brief|detailed|comprehensive|balanced|critical|objective)?\s*"
    r"(?:summary|overview|synthesis|analysis|review|recap)\b"
)

# ── Family 5: Quoted Speech ──
_RE_LONG_QUOTE = re.compile(
    r"['\u2018\u2019][^'\u2018\u2019]{50,}['\u2018\u2019]"
)
_RE_LONG_DOUBLE_QUOTE = re.compile(
    r"[\u201C\"][^\u201C\"]{60,}[\u201D\"]"
)
_RE_ATTRIBUTION = re.compile(
    r"\b(?:said|stated|declared|argued|claimed|asserted|"
    r"announced|warned|cautions?|emphasized|noted|observed|"
    r"highlighted|explained|added|continued|replied|responded)\b"
)

# ── Family 6: Academic Citations ──
_RE_PAREN_CITATION = re.compile(
    r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.|and\s+[A-Z][a-z]+))?,\s*\d{4}[a-z]?\)"
)
_RE_PUBLISHED_IN = re.compile(
    r"\bpublished\s+in\s+(?:the\s+)?(?:journal|proceedings|"
    r"Nature|Science|Cell|The\s+Lancet|NEJM|JAMA|"
    r"Physical\s+Review|IEEE|ACM|Elsevier|Springer)\b"
)
_RE_TITLED = re.compile(
    r"\btitled\s+['\u201C][A-Z][A-Za-z\s:,-]+['\u201D]"
)

# ── Family 7: Statistical Numbers ──
_RE_PERCENTAGE_STAT = re.compile(
    r"\d+(?:\.\d+)?%\s+(?:of\s+the|reduction|increase|drop|rise|"
    r"of\s+patients|of\s+participants|of\s+the\s+population|"
    r"of\s+people|of\s+women|of\s+men|of\s+adults|of\s+children)"
)
_RE_MEASUREMENT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:megajoules|gigawatts|kilowatts|nanometers|"
    r"micrometers|millimeters|centimeters|kilograms|grams|milligrams|"
    r"liters|milliliters|gallons|dollars|euros|pounds|yen|"
    r"cubic\s+meters|per\s+second|per\s+minute|per\s+hour|per\s+day|per\s+year)\b"
)
_RE_RANGE_AS_STAT = re.compile(
    r"\d+(?:\.\d+)?\s*[-]\s*\d+(?:\.\d+)?\s+(?:per|percent|%|of|"
    r"megajoules|kilowatts|dollars|euros|people|patients|"
    r"participants|years|shots|times|episodes|cases|deaths)"
)

# ── Suppression (Anti-Signals) ──
_RE_CODE_FENCE = re.compile(r"```\s*(?:python|java|javascript|rust|go|ruby)")
_RE_KNIGHT_KNAVE = re.compile(
    r"\b(knights?|knaves?|always\s+tells?\s+the\s+truth|always\s+lies?)\b"
)
_RE_MCQ_CHOICES = re.compile(r"^(?:A[.)]|B[.)]|C[.)]|D[.)])", re.MULTILINE)
_RE_MATH_SUBQUESTION = re.compile(
    r"(?:a\)|b\)|c\)|d\)|i\)|ii\)|iii\))\s*(?:Calculate|Find|Solve|Compute)"
)
```

### F. Full scoring function (pseudocode)

```python
def _score_summarization_secondary(prompt: str) -> float:
    """Score a prompt for summarization using secondary signals."""
    s = 0.0
    lower = prompt.lower()
    words = lower.split()
    word_count = len(words)

    # ── Anti-signals: if these fire, definitely not summarization ──
    if _RE_CODE_FENCE.search(prompt):
        return -10.0
    if _RE_KNIGHT_KNAVE.search(lower):
        return -10.0
    if _RE_MCQ_CHOICES.search(prompt):
        return -5.0
    if _RE_MATH_SUBQUESTION.search(prompt):
        return -3.0

    # ── Family 1: Multi-Source ──
    source_count = len(_RE_MULTI_SOURCE.findall(prompt))
    if source_count >= 2:
        s += 3.0
    if _RE_SOURCE_ATTRIB.search(prompt):
        s += 2.0
    if _RE_QUOTED_SOURCE.search(prompt):
        s += 2.0

    # ── Family 2: Document Headers ──
    if _RE_DOC_HEADER.search(prompt):
        s += 3.0
    if _RE_DIPLOMATIC.search(prompt):
        s += 2.5
    if _RE_ALLCAPS_HEADER.search(prompt):
        s += 1.5

    # ── Family 3: News Dateline ──
    if _RE_DATELINE_START.search(prompt):
        s += 3.0
    if _RE_HEADLINE_FORMAT.search(prompt):
        s += 2.0
    news_lead_count = len(_RE_NEWS_LEAD.findall(lower))
    if news_lead_count >= 1 and word_count > 60:
        s += 2.5

    # ── Family 4: Synthesize Instruction ──
    if _RE_SYNTHESIZE.search(prompt):
        s += 4.0
    if _RE_CORE_CAPTURE.search(prompt):
        s += 3.0
    if _RE_PROVIDE_SUMMARY.search(prompt):
        s += 3.0

    # ── Family 5: Quoted Speech ──
    quote_count = len(_RE_LONG_QUOTE.findall(prompt))
    double_quote_count = len(_RE_LONG_DOUBLE_QUOTE.findall(prompt))
    if quote_count + double_quote_count >= 2:
        s += 2.5
    elif quote_count + double_quote_count >= 1 and word_count > 80:
        s += 1.5

    # ── Family 6: Academic Citations ──
    if _RE_PAREN_CITATION.search(prompt):
        s += 2.0
    if _RE_PUBLISHED_IN.search(prompt):
        s += 2.0
    if _RE_TITLED.search(prompt):
        s += 1.0

    # ── Family 7: Statistical Numbers ──
    num_count = len(re.findall(r"\d+(?:\.\d+)?", prompt))
    math_verbs = re.findall(
        r"\b(calculate|compute|solve|equation|formula|derivative|integral|"
        r"algebra|geometry|probability|permutation|combination|factorial|"
        r"sum of|product of|difference of|divided by|quotient|remainder)\b",
        lower
    )
    if _RE_PERCENTAGE_STAT.search(lower):
        s += 1.5
    if _RE_MEASUREMENT.search(lower):
        s += 1.0
    if _RE_RANGE_AS_STAT.search(lower):
        s += 2.0  # Range as statistic → strong summarization signal
        # This also tells us to SUPPRESS math score

    # Numeric density + low math verb density → summarization
    if word_count > 120 and num_count >= 6 and len(math_verbs) == 0:
        s += 2.0
    if word_count > 200 and num_count >= 10 and len(math_verbs) <= 1:
        s += 2.0

    # Length alone = weak signal when combined with any other summarization signal
    if word_count > 150 and s > 0:
        s += 0.5

    return s
```

## Known Edge Cases

### Case 1: News article with range expressions
**Prompt**: "NIF can only conduct a few experiments per day... The laser repetition rate
must increase from a few shots per day to 10-20 per second."
**Problem**: "10-20" → math `num_op`
**Solution**: `_RE_RANGE_AS_STAT` matches "10-20 per second" → boost summarization +2.0

### Case 2: Technical article with code-like words
**Prompt**: "returns True if the model is able to generate a coherent summary of the
input text..."
**Problem**: `return\b` → code_gen +3.0
**Solution**: Check if "return" is followed by "True"/"False"/"None"/a value or by a
prose continuation. In summarization secondary, detect code words in narrative context
and suppress.

### Case 3: Legal document analysis
**Prompt**: "LEGAL BRIEF - PLAINTIFFS' ARGUMENT... The complaint cites that..."
**Problem**: May be classified as logic (analysis/reasoning) or factual
**Solution**: `_RE_DOC_HEADER` + quoted speech detection

### Case 4: Multi-source synthesis without "summarize"
**Prompt**: "SOURCE 1 (UNFCCC...): 'COP28 concluded...' SOURCE 2 (Alliance...): 'We
did not come here...' SOURCE 3 (Carbon Tracker...): '...' Synthesize the three sources
into a balanced summary capturing the core contradictions."
**Problem**: Classified as math (numbers: "$700 million", "2.7 degrees"), logic
(analysis language), or factual (context + question style)
**Solution**: `_RE_SYNTHESIZE` + `_RE_MULTI_SOURCE` + `_RE_SOURCE_ATTRIB` all fire

### Case 5: BBC news (xsum dataset)
**Prompt**: "A 19-year-old man has been charged after a cat was killed and others... "
**Problem**: Too short for length-based signals, no explicit summarization keywords
**Solution**: News lead pattern + absence of any other category signal

### Case 6: Long technical prose (fusion article)
**Prompt**: "On December 5, 2022, scientists at the National Ignition Facility (NIF)
at Lawrence Livermore National Laboratory achieved a historic milestone..."
**Problem**: "1.54 megajoules", "10-20 per second", "2.05 megajoules" → math. Also
"return" → code_gen.
**Solution**: Dateline start + measurement patterns + range-as-statistic + high number
density with zero math verbs

## Integration Points

### 1. `agent/classifier.py` — Add after factual secondary:
```python
# ── Secondary: summarization detector ──
mod = _get_secondary("summarization")
if mod:
    corrected = mod.resolve_summarization(category, prompt, result.get("raw_scores", {}))
    if corrected != category:
        category = corrected
        method = "summarization_secondary"
```

### 2. `agent/hierarchical_classifier.py` — Add analogous block.

### 3. `agent/category_filter.py` — (Optional) Patch `_score_math()` guard:
The math long-text guard (line 178) should also suppress when `num_op` is the
only signal without `explicit` or `word_prob`:

```python
# Current:
if len(words) > 80 and not (explicit or word_prob):
    return s  # num_op bypasses this!

# Fix:
if len(words) > 80 and not (explicit or word_prob or num_op):
    return s  # Now catches range-in-prose scenarios
```

Wait — actually this change would suppress math scoring when num_op fires in long
texts. Let me think... The current logic:
- `num_op` = True if arithmetic operators found (but with some exclusions for ranges)
- If `num_op` is True, s gets +2.0
- Then the guard at line 178: `if len(words) > 80 and not (explicit or word_prob): return s` — this returns `s` early WITHOUT the numeric density boost, but s already has +2.0 from num_op

The problem is that `num_op` is True for "10-20 per second" because:
1. `raw_ops = re.findall(r"\d+\s*([+/*-])\s*\d+", prompt)` finds "10-20" → extracts "-"
2. The second condition checks if `op == "-"` and `prompt.count("-") >= 1` and NOT patent/code/age range patterns
3. "10-20 per second" is NOT excluded by the current guards (they only check `[A-Za-z]+\\d+\\s*-\\s*\\d+` for patent codes, and `\\d+-year|-year-old|aged?\\s+\\d+-` for ages)

So the fix is either:
(a) Add "per [unit]" patterns to the exclusion list in `_score_math`
(b) Add a new exclusion for range + unit-of-measurement patterns

**Recommended**: Both (a) in the primary scorer AND the secondary classifier as a safety net.

## Testing Strategy

1. **Unit tests**: Create `agent/test_secondary_summarization.py` with ~30 prompts:
   - 10 known misclassified summarization prompts (from xsum, claude-code-hard-v1)
   - 10 math prompts (to verify no false positives)
   - 5 code prompts
   - 5 logic prompts

2. **Integration test**: Run the full eval suite before/after to measure summarization
   accuracy improvement.

3. **Regression check**: Verify no new false positives in other categories.

## Appendix: Known Misclassified Prompts (from GPU eval)

| Task ID | Category | Prompt Snippet | Primary Prediction |
|---------|----------|---------------|-------------------|
| q-01aac354 | summarization | "Apple framed its USB-C adoption..." | unknown (likely logic/factual) |
| q-002c3bdf | summarization | "The Permanent Representative..." | "The Permanent Representative..." (restating) |
| q-04b891f1 | summarization | "Study A found moderate red wine consumption..." | unknown (numbers + study reference → math) |
| q-011a76cf | summarization | "Boeing's 2024 safety report touts..." | unknown |
| q-04410ce6 | summarization | "Meta claims improved election interference..." | unknown (3-source synthesis) |
| NIF article | summarization | "On December 5, 2022, scientists at NIF..." | likely math (range + numbers) |
| LEGAL BRIEF | summarization | "LEGAL BRIEF - PLAINTIFFS' ARGUMENT..." | likely logic (analysis language) |
| HEADLINE | summarization | "HEADLINE: 'MIRACLE DRUG CURES ALZHEIMER'S...'" | likely factual (context+question) |
| xsum-* (18 items) | summarization | "A 19-year-old man has been charged..." | unknown (short → factual default) |
