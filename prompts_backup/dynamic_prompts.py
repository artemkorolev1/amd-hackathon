"""
Dynamic System Prompt System for AMD ACT II Hackathon Routing Pipeline.

Assembles per-category, per-complexity system prompts optimized for the
grader's fuzzy-matching evaluation cascade (exact → substring → numeric
tolerance → token overlap). Design derived from:
  - Winning repo analysis (KaananeTaha/AMD-AI-Hackathon, jaeyooniee/track1-hybrid)
  - Project's own caveman_prompts.py, amd-track1/agent/prompts.py, hybrid-token-router/src/prompts.py
  - lablab-Track1/agent/prompts.py (three-tier: local / teacher / api)
  - Grader compatibility analysis (evaluate.py fuzzy_match cascade)
  - Eval ground truth patterns (eval_expected.json checker types)

Key principles:
  1. "Answer:" line on its own → substring match-friendly for the grader
  2. Anti-preamble suffix → first token = the answer (fights model verbosity)
  3. Per-category format enforcement → matches expected checker types
  4. Complexity tier → more scaffolding for complex tasks, terse for simple
  5. Multi-axis features → injected conditionally (creativity/verbosity/structured)
  6. English-only (hackathon requirement)
  7. No preamble, no restating the question, no closing remarks
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Anti-preamble suffix — appended to EVERY system prompt
# ---------------------------------------------------------------------------
# The grader cascade rewards answers where the expected text appears as a
# substring. Every preamble/title/redundant word between the start of the
# answer and the expected token risks a substring match miss.
_ANTI_PREAMBLE = (
    " English only. Start with the answer directly — no greeting, "
    "no 'I will', no 'The user asks', no meta-commentary. "
    "First word = the answer."
)


# ---------------------------------------------------------------------------
# Per-category base prompts (no complexity adjustment)
# ---------------------------------------------------------------------------
# Each entry is a dict with keys for every complexity level.
# The format: "Answer:" prefix is deliberately included for categories
# where an explicit answer label helps the grader's substring match.
# Complexity levels control step visibility and verbosity constraints.

_CATEGORY_PROMPTS: dict[str, dict[str, str]] = {
    # ═══════════════════════════════════════════════════════════════════════
    # CODE GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    # Grader checker: code_tests (executes function, compares outputs)
    # Key: function name must match, signature must match, correct logic
    "code_gen": {
        "low": (
            "Write the requested Python function. "
            "Output ONLY the function inside ```python ... ```. "
            "Preserve the exact function name and signature. "
            "Handle edge cases. No explanation, no docstring, no comments."
        ),
        "medium": (
            "Write the requested Python function. "
            "Output ONLY the function inside ```python ... ```. "
            "Preserve the exact function name and signature. "
            "Handle edge cases (empty input, None, duplicates, type errors). "
            "Add a one-line docstring. No explanatory text outside the code block."
        ),
        "high": (
            "Write the requested Python function. "
            "Output ONLY the function inside ```python ... ```. "
            "Preserve the exact function name and signature. "
            "Handle edge cases thoroughly. "
            "Include a brief docstring and handle all corner cases. "
            "If the spec mentions performance, optimize appropriately. "
            "No explanatory text outside the code block."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # CODE DEBUGGING
    # ═══════════════════════════════════════════════════════════════════════
    # Grader checker: code_tests (extracts function, runs test cases)
    # Key: fix the bug, preserve function name/signature
    "code_debug": {
        "low": (
            "Output ONLY the fully corrected function inside ```python ... ```. "
            "Preserve the original function name and signature. "
            "No description of the bug. No explanation. Just the fixed code."
        ),
        "medium": (
            "Output ONLY the fully corrected function inside ```python ... ```. "
            "Preserve the original function name and signature. "
            "Make sure the fix handles all edge cases the original missed. "
            "No description of the bug. No explanation. Just the fixed code."
        ),
        "high": (
            "Output ONLY the fully corrected function inside ```python ... ```. "
            "Preserve the original function name and signature. "
            "Consider edge cases (off-by-one, empty inputs, type mismatches). "
            "Ensure the fix is minimal and correct. "
            "No description of the bug. No explanation. Just the fixed code."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # MATH (arithmetic & word problems)
    # ═══════════════════════════════════════════════════════════════════════
    # Grader checker: numeric (1% tolerance) — single value
    # Key: numeric output in standard format, no units unless required
    "math": {
        "low": (
            "Solve the math problem. "
            "Output ONLY the final numeric answer on a line starting 'Answer: '. "
            "No units, no explanation, no working. "
            "Use standard decimal format (e.g. 17.5, not 17,5). "
            "Round to the precision implied by the problem. "
            "End with 'Answer: <value>' on its own line."
        ),
        "medium": (
            "Solve the math problem step by step — show brief working "
            "(at most 2-3 steps). "
            "End with 'Answer: <value>' on its own line. "
            "If units are relevant, include them ONLY after the value. "
            "Round to the nearest tenth unless the problem specifies otherwise. "
            "Use standard decimal format."
        ),
        "high": (
            "Solve the math problem step by step — show your reasoning "
            "clearly but concisely (at most 3-4 steps). "
            "Double-check calculations. "
            "End with 'Answer: <value>' on its own line. "
            "Include units if applicable. Round appropriately. "
            "Use standard decimal format."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # LOGIC / DEDUCTIVE REASONING
    # ═══════════════════════════════════════════════════════════════════════
    # Grader checker: substring match (expected answer is short: name/animal/tool)
    # Key: final answer must be a short, unambiguous phrase
    "logic": {
        "low": (
            "Solve the logic puzzle. "
            "Show a single step of reasoning, then "
            "end with 'Answer: <conclusion>' on its own line. "
            "Keep the conclusion short — a name, item, or single word."
        ),
        "medium": (
            "Solve the logic puzzle step by step — show your reasoning "
            "in 2-3 clear steps. "
            "End with 'Answer: <conclusion>' on its own line. "
            "Keep the conclusion short and precise."
        ),
        "high": (
            "Solve the logic puzzle carefully. "
            "Show reasoning step by step — use a table, grid, or "
            "deductive chain if needed. "
            "Verify every condition is satisfied. "
            "End with 'Answer: <conclusion>' on its own line. "
            "Keep the conclusion short and unambiguous."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # FACTUAL KNOWLEDGE / QA
    # ═══════════════════════════════════════════════════════════════════════
    # Grader checker: contains_all (multi-part answers common)
    # Key: include ALL requested facts, use exact names/numbers
    "factual": {
        "low": (
            "Answer the question directly. "
            "Address every part of multi-part questions. "
            "Keep the answer under 50 words. "
            "Use exact names, dates, and numbers. "
            "No preamble, no closing."
        ),
        "medium": (
            "Answer the question clearly and completely. "
            "Address every part of multi-part questions — if the question "
            "asks for two facts, include both. "
            "Keep the answer under 120 words. "
            "Use exact names, dates, and numbers. "
            "No preamble, no closing."
        ),
        "high": (
            "Answer the question thoroughly and accurately. "
            "Address every sub-part explicitly. "
            "Keep the answer under 200 words. "
            "Use exact names, dates, and numbers. "
            "If explaining a concept, be precise but concise. "
            "No preamble, no closing."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # SENTIMENT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    # Grader checker: label match (positive/negative/neutral/mixed)
    # Key: exact label first, then optional justification
    "sentiment": {
        "low": (
            "Classify the sentiment. "
            "Be careful — do NOT default to 'positive'. "
            "Sarcasm, dismissiveness, and hedging are NEGATIVE or NEUTRAL, not positive. "
            "Re-read your answer: are you sure? Sarcasm, irony, and dismissiveness are NEGATIVE. "
            "If the text expresses ANY negative emotion (anger, disappointment, "
            "frustration, sadness), classify as negative even if there's "
            "some positive language. "
            "Default to negative when uncertain about sarcasm. "
            "Output EXACTLY one word: positive, negative, neutral, or mixed. "
            "No explanation, no preamble."
        ),
        "medium": (
            "Classify the sentiment carefully. "
            "Watch for sarcasm, dismissiveness, and hedging — these are NOT positive. "
            "Re-read your answer: are you sure? Sarcasm, irony, and dismissiveness are NEGATIVE. "
            "If the text expresses ANY negative emotion (anger, disappointment, "
            "frustration, sadness), classify as negative even if there's "
            "some positive language. "
            "Default to negative when uncertain about sarcasm. "
            "Start with the label on its own line: Label: positive|negative|neutral|mixed. "
            "Then one short sentence of justification citing specific phrases."
        ),
        "high": (
            "Classify the sentiment carefully. "
            "Watch for sarcasm, dismissiveness, hedging, and double negatives. "
            "Start with the label on its own line: Label: positive|negative|neutral|mixed. "
            "Then a brief justification (2-3 sentences) referencing specific "
            "words or phrases that support your classification. "
            "Consider subtlety: sarcasm, double negatives, mixed signals."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # NAMED ENTITY RECOGNITION
    # ═══════════════════════════════════════════════════════════════════════
    # Grader checker: contains_all (entity text + type)
    # Key: structured output with entity and type, covering ALL entities
    "ner": {
        "low": (
            "Extract all named entities. "
            "Group entities by type and output as: CATEGORY: value1, value2; CATEGORY: value3. "
            "Use precise category labels based on what the text contains "
            "(e.g. PERSON, ORGANIZATION, LOCATION, DATE, GENE, DISEASE, "
            "TICKER, MONETARY, PERCENTAGE, DRUG, LEGISLATION). "
            "Only include entities that explicitly appear in the text. "
            "No preamble, no commentary, no sentences."
        ),
        "medium": (
            "Extract all named entities exhaustively. "
            "Group entities by type and output as: CATEGORY: value1, value2; CATEGORY: value3. "
            "Use precise category labels based on what the text contains "
            "(e.g. PERSON, ORGANIZATION, LOCATION, DATE, GENE, DISEASE, "
            "TICKER, MONETARY, PERCENTAGE, DRUG, LEGISLATION, TITLE). "
            "Cover every named entity — don't skip any. "
            "No preamble, no commentary, no sentences. "
            "Output ONLY the structured category groupings."
        ),
        "high": (
            "Extract all named entities exhaustively. "
            "Group entities by type and output as: CATEGORY: value1, value2; CATEGORY: value3. "
            "Use precise category labels based on what the text contains "
            "(e.g. PERSON, ORGANIZATION, LOCATION, DATE, GENE, DISEASE, "
            "TICKER, MONETARY, PERCENTAGE, DRUG, LEGISLATION, TITLE, "
            "PROTEIN, CELL_LINE, ANATOMICAL, DOSAGE). "
            "Cover every named entity. Be careful with ambiguous cases. "
            "No preamble, no commentary. Output ONLY the structured groupings."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARIZATION
    # ═══════════════════════════════════════════════════════════════════════
    # Grader checker: sentence_count or word_max
    # Key: strictly obey the stated length constraint
    "summarization": {
        "low": (
            "Summarize the text. "
            "Strictly obey ANY length constraint stated in the prompt "
            "(sentence count, word count). If no length is given, "
            "output at most 2 sentences. "
            "No preamble, no 'Here is a summary'. "
            "Output ONLY the summary text."
        ),
        "medium": (
            "Summarize the text. "
            "Strictly obey ANY length constraint stated in the prompt "
            "(sentence count, word count, bullet count). "
            "If no length is given, output at most 3 sentences. "
            "Capture all key points. "
            "No preamble, no 'Here is a summary'. "
            "Output ONLY the summary text."
        ),
        "high": (
            "Summarize the text. "
            "Strictly obey ANY length constraint stated in the prompt "
            "(sentence count, word count, bullet count). "
            "If no length is given, output at most 3 sentences. "
            "Capture all key points while maintaining factual accuracy. "
            "Preserve important names, numbers, and relationships. "
            "No preamble, no 'Here is a summary'. "
            "Output ONLY the summary text."
        ),
    },
}


# ---------------------------------------------------------------------------
# Multi-axis feature injectors
# ---------------------------------------------------------------------------
# These are conditionally appended when the corresponding feature score
# exceeds a threshold from the multi-axis feature extractor.

_FEATURE_INJECTIONS: dict[str, dict[str, str]] = {
    "creativity": {
        "low": "",
        "medium": " Be creative but stay factual where the task requires accuracy.",
        "high": " Feel free to be creative and original. Avoid clichés.",
    },
    "verbosity": {
        "low": "",
        "medium": " Be concise — use the minimum words needed.",
        "high": " Be extremely terse — one sentence or less when possible.",
    },
    "structured_output": {
        "low": "",
        "medium": " Use clear structure like bullet points or numbered lists where helpful.",
        "high": (
            " Output in a strict structured format. Use JSON or clearly labeled "
            "sections. Every part of the answer must be explicitly labeled."
        ),
    },
    "multi_step": {
        "low": "",
        "medium": " Think step by step before answering.",
        "high": (
            " Reason carefully through each step before producing the final answer. "
            "Break the problem into sub-problems and solve each one."
        ),
    },
}


# ---------------------------------------------------------------------------
# Complexity signal map — maps complexity scores (0 = simple, 1 = complex)
# to the three levels used in the category prompts.
# ---------------------------------------------------------------------------

def _complexity_level(score: float) -> str:
    """Map a 0-1 complexity score to 'low', 'medium', or 'high'."""
    if score < 0.3:
        return "low"
    elif score < 0.7:
        return "medium"
    else:
        return "high"


NER_ONE_SHOT_EXAMPLE = (
    "Example output formats for entity extraction:\n\n"
    "Text: \"WNT signaling activates beta-catenin, which translocates to the nucleus "
    "and drives medulloblastoma growth at Cold Spring Harbor Laboratory.\"\n"
    "Output: GENE: WNT, beta-catenin; DISEASE: medulloblastoma; "
    "ORGANIZATION: Cold Spring Harbor Laboratory\n\n"
    "Text: \"Acme Corporation announced Q4 earnings of $2.3B on January 15, 2024, "
    "led by CEO Rebecca Nguyen in Delaware.\"\n"
    "Output: ORGANIZATION: Acme Corporation; MONEY: $2.3B; DATE: January 15, 2024; "
    "PERSON: Rebecca Nguyen; LOCATION: Delaware\n\n"
    "Text: \"The European Commission fined Apple €1.8 billion under the Digital Markets Act "
    "for anti-steering practices on March 4, 2024.\"\n"
    "Output: ORGANIZATION: European Commission, Apple; MONEY: €1.8 billion; "
    "LEGISLATION: Digital Markets Act; DATE: March 4, 2024\n\n"
    "Text: \"Dr. Sarah Chen of Mass General Brigham presented stage 3 clinical trial data "
    "for lecanemab at the Alzheimer's Association International Conference.\"\n"
    "Output: PERSON: Sarah Chen; ORGANIZATION: Mass General Brigham, Alzheimer's Association "
    "International Conference; DRUG: lecanemab; DISEASE: Alzheimer's\n\n"
    "Text: \"The Federal Reserve raised interest rates by 25 basis points to 5.5% on July 26, 2023, "
    "citing persistent inflation.\"\n"
    "Output: ORGANIZATION: Federal Reserve; PERCENTAGE: 25 basis points, 5.5%; "
    "DATE: July 26, 2023; CONCEPT: persistent inflation\n\n"
    "Now extract entities from the following text. Output the same CATEGORY: value format."
)

SENTIMENT_EXAMPLES = (
    "Example sentiment classifications with edge cases:\n\n"
    "Text: \"Oh great, another meeting. Just what I needed.\"\n"
    "Label: negative\n"
    "Reason: Sarcastic — 'great' is clearly ironic given the context.\n\n"
    "Text: \"The product works fine, I suppose. It gets the job done.\"\n"
    "Label: neutral\n"
    "Reason: Hedging — 'fine', 'I suppose' indicate reluctance, not enthusiasm.\n\n"
    "Text: \"I absolutely love waiting 45 minutes for customer service. Best day ever.\"\n"
    "Label: negative\n"
    "Reason: Sarcastic — positive words used to express frustration.\n\n"
    "Text: \"The interface is clean and responsive, but the lack of dark mode is disappointing.\"\n"
    "Label: mixed\n"
    "Reason: Contains both positive ('clean', 'responsive') and negative ('disappointing').\n\n"
    "Text: \"The movie was... interesting. I've definitely seen worse, I guess.\"\n"
    "Label: neutral\n"
    "Reason: Hedging — vague language expresses neither strong praise nor criticism.\n\n"
    "Now classify the following text. Output EXACTLY one word: positive, negative, neutral, or mixed."
)

MATH_EXAMPLES = (
    "Example multi-step math problems with verification:\n\n"
    "Problem: Two pipes A and B can fill a tank in 12 hours and 16 hours respectively. "
    "Pipe B alone is opened for first 4 hours, then pipe A is also opened. "
    "How long total to fill the tank?\n"
    "Step 1: Pipe B fills 1/16 per hour. In 4 hours: 4/16 = 1/4 of tank filled.\n"
    "Step 2: Remaining: 1 - 1/4 = 3/4 of tank.\n"
    "Step 3: Both pipes together: 1/12 + 1/16 = 4/48 + 3/48 = 7/48 per hour.\n"
    "Step 4: Remaining time: (3/4) / (7/48) = (3/4) * (48/7) = 144/28 = 36/7 ≈ 5.14 hours.\n"
    "Step 5: Total: 4 + 5.14 = 9.14 hours. Verification: (4/16) + (36/7)*(7/48) = 1/4 + 3/4 = 1 ✓\n"
    "Answer: 9.14\n\n"
    "Problem: A shopkeeper buys 50 kg of sugar at $2 per kg and 30 kg at $3 per kg. "
    "He mixes them and sells at $4 per kg. What is his profit percentage?\n"
    "Step 1: Total cost = 50*2 + 30*3 = 100 + 90 = $190.\n"
    "Step 2: Total weight = 50 + 30 = 80 kg.\n"
    "Step 3: Total revenue = 80 * 4 = $320.\n"
    "Step 4: Profit = 320 - 190 = $130.\n"
    "Step 5: Profit percentage = (130/190) * 100 = 68.4%.\n"
    "Answer: 68.4\n\n"
    "Now solve the following problem. Show steps, verify your calculation, end with 'Answer: <value>'."
)

# ---------------------------------------------------------------------------
# Merged prompts — used when S2 is uncertain between two categories (scores close)
# ---------------------------------------------------------------------------
# These combine instructions for two categories so that S2 misclassification
# between them doesn't hurt the answer. Key confusion pairs from 300-set data:
#   summarization x math (6/6 fail), logic x factual (4/9), logic x math (3/5)
MERGE_PROMPTS: dict[str, dict[str, str]] = {
    "reasoning": {
        # Covers logic + math + factual — the three most confused categories
        # where S2 uncertainty costs real accuracy
        "low": (
            "Solve the problem directly. If it involves numbers, "
            "output the value after 'Answer: '. If it's a puzzle or deduction, "
            "output the conclusion after 'Answer: '. "
            "If it's factual knowledge, answer directly. "
            "No preamble, no explanation. Just the answer."
        ),
        "medium": (
            "Solve the problem step by step. Show brief reasoning (2-3 steps). "
            "End with 'Answer: <value_or_conclusion>' on its own line. "
            "Use exact names, numbers, and dates where applicable. "
            "No preamble, no closing."
        ),
        "high": (
            "Solve the problem carefully step by step. "
            "Show reasoning (3-4 steps). Double-check calculations and deductions. "
            "End with 'Answer: <value_or_conclusion>' on its own line. "
            "Use exact names, numbers, and dates. "
            "No preamble, no closing."
        ),
    },
    "logic_deduction": {
        # Pure-logic merge key — triggered when primary_category == "logic".
        # Uses deductive-chain language, knight/knave framing, syllogism patterns,
        # and inference steps, unlike the generic "reasoning" key.
        "low": (
            "Deduce the answer using logic. For knight/knave puzzles, syllogisms, "
            "or deductive puzzles, infer the conclusion from the given premises. "
            "Output the conclusion after 'Answer: '. "
            "No preamble, no explanation. Just the conclusion."
        ),
        "medium": (
            "Solve the logic puzzle step by step. Deduce the answer from the premises — "
            "use inferential reasoning, eliminate contradictions, and verify each "
            "condition. End with 'Answer: <conclusion>' on its own line. "
            "Keep the conclusion short and precise."
        ),
        "high": (
            "Solve the logic puzzle carefully using deductive reasoning. "
            "Use a truth table, grid, or syllogistic chain to trace through each "
            "condition. Eliminate impossible cases. Verify every condition is "
            "satisfied before concluding. "
            "End with 'Answer: <conclusion>' on its own line. "
            "Keep the conclusion short and unambiguous."
        ),
    },
    "summarization_math": {
        # Covers summarization x math confusion — the most damaging single pair (6/6 fail)
        "low": (
            "If this is a math or computation problem, "
            "output ONLY the numeric answer on a line starting 'Answer: '. "
            "If this is a text to summarize, "
            "output ONLY the summary obeying any length constraint. "
            "No preamble, no explanation."
        ),
        "medium": (
            "Determine whether this is a math problem or a summarization task. "
            "If math: show brief working (2-3 steps), end with 'Answer: <value>'. "
            "If summarization: strictly obey any length constraint, capture key points. "
            "No preamble, no closing."
        ),
        "high": (
            "Determine whether this is a math problem or a summarization task. "
            "If math: show reasoning, double-check, end with 'Answer: <value>'. "
            "If summarization: obey length constraints, capture all key points "
            "while maintaining factual accuracy. "
            "No preamble, no closing."
        ),
    },
}


def build_merged_prompt(
    primary_category: str,
    secondary_category: str = "",
    complexity_score: float = 0.5,
    custom_instructions: str = "",
) -> str:
    """Build a merged prompt when S2 is uncertain between two categories.

    Uses a pre-defined merged prompt for known confusion pairs, or falls back
    to the primary category prompt if no merge is defined.
    """
    pair = (primary_category, secondary_category)
    reverse_pair = (secondary_category, primary_category)

    # Known merge keys
    # When primary_category == "logic", use logic-specific deduction language
    # instead of the generic "reasoning" key (fixes logic accuracy regression).
    merge_key = ""
    if primary_category == "logic" and ("math" in pair or "factual" in pair):
        merge_key = "logic_deduction"
    elif "math" in pair and "logic" in pair:
        merge_key = "reasoning"
    elif "math" in pair and "factual" in pair:
        merge_key = "reasoning"
    elif "logic" in pair and "factual" in pair:
        merge_key = "reasoning"
    elif "summarization" in pair and "math" in pair:
        merge_key = "summarization_math"

    if merge_key:
        level = _complexity_level(complexity_score)
        base = MERGE_PROMPTS[merge_key].get(level, MERGE_PROMPTS[merge_key]["medium"])
        parts = []
        if custom_instructions:
            parts.append(custom_instructions.strip())
        parts.append(base)
        parts.append(_ANTI_PREAMBLE)
        return " ".join(parts).strip()

    # No merge defined — fall back to single-category prompt
    return build_system_prompt(
        category=primary_category,
        complexity_score=complexity_score,
        custom_instructions=custom_instructions,
    )


# ---------------------------------------------------------------------------
# Assembly function
# ---------------------------------------------------------------------------

def build_system_prompt(
    category: str,
    complexity_score: float = 0.5,
    feature_scores: Optional[dict[str, float]] = None,
    custom_instructions: Optional[str] = None,
) -> str:
    """
    Build a dynamic system prompt for the routing pipeline.

    Args:
        category: One of the 8-way categories (code_gen, code_debug, math,
                  logic, factual, sentiment, ner, summarization).
        complexity_score: Float 0.0-1.0 from per-category complexity classifier.
                          Controls step visibility and verbosity constraints.
        feature_scores: Dict of multi-axis feature scores (0.0-1.0) from
                        Stage 1 feature extraction. Supported keys:
                        creativity, verbosity, structured_output, multi_step.
        custom_instructions: Optional string to prepend (e.g., few-shot examples).

    Returns:
        Assembled system prompt string suitable for the LLM system message.
    """
    if feature_scores is None:
        feature_scores = {}

    # Resolve category key
    cat_key = category
    if category not in _CATEGORY_PROMPTS:
        # Fallback to factual for unknown categories
        cat_key = "factual"

    # Get complexity level
    level = _complexity_level(complexity_score)
    base = _CATEGORY_PROMPTS[cat_key].get(level, _CATEGORY_PROMPTS[cat_key]["medium"])

    # Collect feature injections
    injections = []
    for feat, level_map in _FEATURE_INJECTIONS.items():
        feat_score = feature_scores.get(feat, 0.0)
        if feat_score > 0.5:
            feat_level = _complexity_level(feat_score)
            injection = level_map.get(feat_level, "")
            if injection:
                injections.append(injection)

    # Assemble
    parts = []
    if custom_instructions:
        parts.append(custom_instructions.strip())
    parts.append(base)
    parts.extend(injections)
    parts.append(_ANTI_PREAMBLE)

    return " ".join(parts).strip()


def build_solver_messages(
    category: str,
    task: str,
    complexity_score: float = 0.5,
    feature_scores: Optional[dict[str, float]] = None,
    custom_instructions: Optional[str] = None,
    deterministic_hint: Optional[str] = None,
) -> list[dict[str, str]]:
    """
    Build messages list for direct LLM solver calls.

    Returns [system_message, user_message] ready for the Fireworks/Ollama API.
    """
    system = build_system_prompt(
        category=category,
        complexity_score=complexity_score,
        feature_scores=feature_scores,
        custom_instructions=custom_instructions,
    )

    # If a deterministic hint is available, prepend it to the user prompt
    user_content = task
    if deterministic_hint:
        # Place the hint BEFORE the task so the model sees it while processing
        user_content = f"Hint: {deterministic_hint}\n\n{task}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Max tokens per category (from proven gate-passing configs)
# ---------------------------------------------------------------------------
# Grounded in the hybrid-token-router's SPEC and the project's caveman_prompts.py.
# Simple tasks need fewer tokens; complex tasks need more headroom.

MAX_TOKENS: dict[str, int] = {
    "code_gen": 250,
    "code_debug": 200,
    "math": 200,
    "logic": 200,
    "factual": 120,
    "sentiment": 60,
    "ner": 120,
    "summarization": 120,
}

DEFAULT_MAX_TOKENS = 300


def get_max_tokens(category: str, complexity_score: float = 0.5) -> int:
    """Get the max_tokens budget, adjusted upward for complex tasks."""
    base = MAX_TOKENS.get(category, DEFAULT_MAX_TOKENS)
    if complexity_score > 0.7:
        # Complex tasks get 30% more headroom
        return int(base * 1.3)
    elif complexity_score > 0.3:
        # Medium tasks get 10% more
        return int(base * 1.1)
    return base


# ---------------------------------------------------------------------------
# Stop sequences per category
# ---------------------------------------------------------------------------
STOP_SEQUENCES: dict[str, list[str]] = {
    "code_gen": ["```\n\n", "\n\n```"],
    "code_debug": ["```\n\n", "\n\n```"],
    "math": ["\n\n", "Question:"],
    "logic": ["\n\n", "Question:"],
    "factual": ["\n\n", "Question:"],
    "sentiment": ["\n\n"],
    "ner": ["\n\n"],
    "summarization": ["\n\n"],
}

DEFAULT_STOP: list[str] = ["\n\n"]


def get_stop_sequences(category: str) -> list[str]:
    """Get stop sequences for a category to cut off generation early."""
    return STOP_SEQUENCES.get(category, DEFAULT_STOP)


# ---------------------------------------------------------------------------
# Predefined prompt configs (for quick-access by the pipeline)
# ---------------------------------------------------------------------------
# These are ready-to-use config dicts that the router can reference by name.
# Each maps category -> (prompt, max_tokens, stop)

PROMPT_TABLE: dict[str, dict[str, tuple[str, int, list[str]]]] = {}

for cat_name, cat_data in _CATEGORY_PROMPTS.items():
    PROMPT_TABLE[cat_name] = {}
    for complexity in cat_data:
        prompt_text = cat_data[complexity] + _ANTI_PREAMBLE
        max_tok = MAX_TOKENS.get(cat_name, DEFAULT_MAX_TOKENS)
        # Scale up for complex
        if complexity == "high":
            max_tok = int(max_tok * 1.3)
        elif complexity == "medium":
            max_tok = int(max_tok * 1.1)
        PROMPT_TABLE[cat_name][complexity] = (
            prompt_text,
            max_tok,
            STOP_SEQUENCES.get(cat_name, DEFAULT_STOP),
        )


def lookup_prompt_config(
    category: str, complexity: str = "medium"
) -> tuple[str, int, list[str]]:
    """Quick lookup in the prompt table.

    Returns (system_prompt, max_tokens, stop_sequences).
    Falls back to factual/medium for unknown categories.
    """
    cat_data = PROMPT_TABLE.get(category)
    if cat_data is None:
        cat_data = PROMPT_TABLE["factual"]
    config = cat_data.get(complexity)
    if config is None:
        config = cat_data["medium"]
    return config
